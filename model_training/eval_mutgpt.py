# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import torch
from functools import partial

from typing import Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.training import evaluate_mutgpt,setup_model_and_optimizer
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
)
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from utils_mutation import *
from utils import *
import random

stimer = StragglerDetector()
data_dir_prefix=""

def print_file(message,f):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, file=f)
    else:
        print(message, file=f)
def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    vdict={0:184531,1:150210,2:191210,3:192710}
    args = get_args()
    vsize=vdict[args.version]
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else: # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vsize,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None 
    
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)
    #print("rank(step1):",mpu.get_tensor_model_parallel_rank(), batch.values())
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)
    #print("rank(step2):",mpu.get_tensor_model_parallel_rank(), batch.values())
    if batch is None:
        return None
    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    # add a term
    vdict={0:184531,1:150210,2:191210,3:192710}
    args.padded_vocab_size=vdict[args.version]

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])
    #print(loss,total_tokens,loss/total_tokens)
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def compute_accuracy(output_tensor,labels,loss_mask,ref_seq,tokens,anno,trans_table,version=0):
    method_ids=['acc_count','h10_count','h100_count','mixed_count','count']
    if version==0:
        orange={0:[34406,183921],1:[1,29899]}
        embed_length=12
    if version==1:
        orange={0:[1,149515]}
        embed_length=1
    if version==2:
        orange={1:[41000,190515],2:[1,29904],3:[29904,41000],4:[29904,42000]}
        embed_length=8
    if version==3:
        orange={1:[42500,192015],2:[1,29904],3:[29904,29924],4:[29924,42000],5:[42000,42500],6:[29924,42000],7:[42000,42500]} 
        embed_length=14
   
    for i in range(4,labels.shape[1]):
        if loss_mask[0,i]>0:
            if labels[0,i]>0:
                ids=(i-4)%embed_length
                output_tensor[:,i,orange[ids][0]:orange[ids][1]+1]+=1000
    
    sorted_ids=torch.topk(output_tensor,1000,dim=2,sorted=True).indices
    output_tensor=torch.nn.functional.softmax(output_tensor,dim=2)

    ret_tensor=torch.zeros([5,8],device=output_tensor.device,requires_grad=False)
    new_seq=copy.deepcopy(ref_seq)
    for i in range(labels.shape[1]):
        ids=(i-4)%embed_length
        if loss_mask[0,i]>0:
            if labels[0,i]>0:
                
                if ids in orange:
                    
                    ret_tensor[4,ids]+=1
                        
                    if labels[0,i]==sorted_ids[0,i,0]:
                        ret_tensor[0:3,ids]+=1
                    else:  
                            
                        j=1
                        find=True
                        while j<100 and find:
                            if labels[0,i]==sorted_ids[0,i,j]:
                                ret_tensor[2,ids]+=1
                                if j<10:
                                    ret_tensor[1,ids]+=1
                                    find=False
                            j+=1
        if version>=2 :
            if tokens[0,i]>0 and ids==embed_length/2+1 and i>=5:
                mut,new_seq=check_mutation(tokens[0,i].cpu().item(),new_seq,version=version)
            if loss_mask[0,i]>0 and labels[0,i]>0 and ids==1:
                scores={}
                for j in range(len(sorted_ids[0,i])):
                    lb=sorted_ids[0,i,j].cpu().item()
                    sc=0
                    numk=0
                    mut,nseq=check_mutation(lb,new_seq,version=version)
                    if int(mut[1:-1])<=29903:
                        encoded_seq,nref=mutation_encoding(new_seq,mut,anno,trans_table,version=version)
                        #print(encoded_seq)
                        if encoded_seq is not None:
                            for k in range(len(encoded_seq)):
                                if encoded_seq[k]>0:
                                    try:
                                        sc+=10+torch.log(output_tensor[0,i+k,int(encoded_seq[k])])
                                    except:
                                        print(k,encoded_seq)
                                    numk+=1
                                else:
                                    sc+=7.5
                            if numk>0:
                                final=sc
                                scores[lb]=final
                max_item = max(scores, key=scores.get)
                #print(max_item,labels[0,i])
                if labels[0,i]==max_item:
                    ret_tensor[3,ids]+=1
        
        if version==1:
            if i>=5:
                mut,new_seq=check_mutation(tokens[0,i].cpu().item(),new_seq,version=version)
            if loss_mask[0,i]>0 and labels[0,i]>0:
                mut,nseq=check_mutation(labels[0,i].cpu().item(),new_seq,version=version)
                lbmts=get_mutation(new_seq,int(mut[1:-1]),mut[-1],anno,trans_table)
                has_s=False
                if lbmts is not None:
                    for item in lbmts:
                        if 'S:' in item:
                            has_s=True
                            smut=item
                if has_s:
                    rk=0
                    isend=False
                    ret_tensor[4,1]+=1
                    for j in range(len(sorted_ids[0,i])):
                        if not(isend):
                            lb=sorted_ids[0,i,j].cpu().item()
                            sc=0
                            numk=0
                            mut,nseq=check_mutation(lb,new_seq,version=version)
                            muts=get_mutation(new_seq,int(mut[1:-1]),mut[-1],anno,trans_table)
                            isend=False
                            if muts is not None:
                                for mt in muts:
                                    if 'S:' in mt :
                                        rk+=1
                                        #print(mt,smut,mt==smut)
                                        if mt==smut:
                                            isend=True
                                            if rk==1:
                                                ret_tensor[0,1]+=1
                                            if rk<=10:
                                                ret_tensor[1,1]+=1
                                        if rk>=10:
                                            isend=True
                    
                


                            
    return ret_tensor


def check_mutation(mut,ref_sequence,version=0):
    vdict={0:34406,1:1,2:41000,3:42500}
    vnum=vdict[version]
    mut=mut-vnum
    pos=mut//5+1
    targetdict={0:'A',1:"T",2:"C",3:'G',4:'-'}
    target_nuc=targetdict[mut%5]
    try:
        origin_nuc=ref_sequence[pos-1]
    except:
        #print(mut,pos,version)
        return target_nuc+str(mut)+target_nuc,ref_sequence
    ref_sequence=ref_sequence[:pos-1]+target_nuc+ref_sequence[pos:]
    mut=origin_nuc+str(pos)+target_nuc
    return mut,ref_sequence

def forward_step(data_iterator, ref_seq,anno,trans_table,model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    
    params = get_batch(
            data_iterator)
    if params is not None:
        tokens, labels, loss_mask, attention_mask, position_ids,weight=params
    else:
        olt=torch.zeros([5,8]).cuda(non_blocking=True)
        #print(olt)
        return olt
    
    timers('batch-generator').stop()
     # 0: nothing 
    # pos: 1-29903 
    # 29904-29923: A->T  A->C  A->G  T->A T->C  T->G  C->A C->T C->G G->A G->T G->C A->- T->- C->- G->-
    # 29924-29935 : annos   (Orf1a, Orf1b, S, Orf3a, E,M, Orf6, Orf7a, Orf7b, Orf8, N, Orf10 )
    # 29936-29942 : overlap_annos (Orf0, Orf3b, Orf3c, Orf3d, Orf3d-2, Orf9b, Orf9c)
    # 29943-33943 : position of related 1-4401
    # 33944-34405 : 21*22=462 transition map
    # 34406-183920 : 29903*5=149515 mutation profiles
    with stimer:
        no_loc=False
        if no_loc:       
            tokens[:,:2]=0

        output_tensor = model(tokens, position_ids, attention_mask)
        #print("get answer")
        ret_tensor=compute_accuracy(output_tensor,labels,loss_mask,ref_seq,tokens,anno,trans_table,version=args.version)
        #print(ret_tensor)
        nummut=ret_tensor[4][0]
        nums=ret_tensor[4][1]
    
    if nummut>0:
        ret_tensor[:,2]=ret_tensor[:,0]/nummut
        ret_tensor[:,0]=ret_tensor[:,0]*weight/nummut
        ret_tensor[3,0]=weight*weight*0.001
    
    if nums>0:
        ret_tensor[:,3]=ret_tensor[:,1]/nums
        ret_tensor[:,1]=ret_tensor[:,1]*weight/nums
        ret_tensor[3,1]=weight*weight*0.001
    return ret_tensor
    
     

def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.mock_data:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds




def evaluate(forward_step_func,
             data_iterator,
             model_provider,
             verbose=False):
    """Evaluation."""

    args = get_args()
    timers = get_timers()
    variant_mutation_dic={}
    anno=read_designation(variant_mutation_dic)
    trans_table=read_table()
    model_type=ModelType.encoder_or_decoder
    timers('evaluate', log_level=0).start(barrier=True)
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
        model_provider, model_type)
    print("finished loading model")
    # Turn on evaluation mode which disables dropout.
    

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]

    model.eval()
    print('finished evaluation mode')
    # make validation batch size independent from training batch size
    eval_batch_size = args.global_batch_size
    eval_num_microbatches = eval_batch_size // \
        (args.micro_batch_size * args.data_parallel_size)
    print('eval batches:',eval_num_microbatches)

    with torch.no_grad():
        iteration = 0
        ref_seq=read_ref()
        print_rank_0(f'Evaluating on {args.eval_iters * eval_batch_size} samples')
        stop_iter=True
        while stop_iter:
            iteration += 1
            print_rank_0(f'Evaluating iter {iteration}')
            
            for i in range(eval_num_microbatches):

                ret_tensor = forward_step(
                data_iterator,ref_seq,anno,trans_table,
                model=model)
                #print(i,ret_tensor)
                
                if i==0:
                    total_tensor=ret_tensor
                else:
                    if ret_tensor[4][0]>=0.1:
                        total_tensor=total_tensor+ret_tensor
                
            torch.distributed.barrier()
            torch.distributed.all_reduce(total_tensor,group=mpu.get_data_parallel_group())
            print_rank_0(total_tensor) 
            if total_tensor[4][0]>=0.1:
                iter_result=total_tensor[0:4]/total_tensor[4:5]
                print_rank_0("iteration "+str(iteration)+':')
                print_rank_0(iter_result)
            else:
                stop_iter=False

            if iteration==1:
                overall_total_tensor=total_tensor*0.001
            else:
                overall_total_tensor=overall_total_tensor+total_tensor*0.001
                print_rank_0(overall_total_tensor)
           

            #print("loss dicts:",loss_dicts)
            

            # Empty unused memory
            if args.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            args.consumed_valid_samples += eval_batch_size
    #print_rank_0(overall_total_tensor)
    overall_total_tensor[0:4]/=overall_total_tensor[4:5]       

    # Move model back to the train mode.
    print_rank_0("final result")
    print_rank_0(overall_total_tensor)
    extracted_tensor=overall_total_tensor.cpu().numpy()
    #print_rank_0(f'nuc top 1:{extracted_tensor[0][0].4f}%, ')
    timers('evaluate').stop()
    timers.log(['evaluate'])

    return extracted_tensor

if __name__ == "__main__":

    # Temporary for transition to core datasets
    initialize_megatron(extra_args_provider=None,
                        args_defaults={})
    args=get_args()
    version=args.version

    train_valid_test_datasets_provider.is_distributed = True
    date=args.valid_dataset
    data_dir=data_dir_prefix+date
    variants_encoding_dict=read_variant_table(data_dir+'/base_variants_v'+str(version)+'.json')
    valid_date=args.valid_start_date

    
    date_valid=[valid_date,date]
    if args.valid_month!=None:
        date_valid=[args.valid_month+'-01',args.valid_month+'-31']
        
    country_list=read_country_list()
    eval_iterator=data_iterator_sample(data_dir+'/shuffled_seqs_v'+str(version),variants_encoding_dict,date_range=date_valid,total_ranks=8,
    country_id=None,version=version,evaluation=True)
    #test_iterator=data_iterator_sample(data_dir+'/shuffled_seqs_weight',variants_encoding_dict,test_list)

    extracted_tensor=evaluate(forward_step,
        eval_iterator,
        model_provider
    )
    if torch.distributed.get_rank() == 0:
        f=open(args.output_file,'w')
        print(extracted_tensor, file=f)
        print(f"Nuc top 1:{extracted_tensor[0][0]:.4f}, Nuc top 10:{extracted_tensor[1][0]:.4f}, Nuc top 100:{extracted_tensor[2][0]:.4f}", file=f)
        print(f"Spike top 1:{extracted_tensor[0][1]:.4f}, Spike top 10:{extracted_tensor[1][1]:.4f}", file=f)
        print(f"Raw Nuc top 1:{extracted_tensor[0][2]:.4f}, Nuc top 10:{extracted_tensor[1][2]:.4f}, Nuc top 100:{extracted_tensor[2][2]:.4f}", file=f)
        print(f"Raw Spike top 1:{extracted_tensor[0][3]:.4f}, Spike top 10:{extracted_tensor[1][3]:.4f}", file=f)
        f.close()
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
from megatron.training import evaluate_mutgpt,read_model
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
from megatron.training.arguments import parse_args, validate_args
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from utils_mutation import *
from utils import *
import random
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

stimer = StragglerDetector()

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
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
        vdict={0:184531,1:150210,2:191210,3:192710}
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vdict[args.version],
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
        #print(model)
    return model


def get_batch(data_iterator):
    """Generate a batch."""
    data=next(data_iterator)
    batch = {
           'tokens': data["tokens"].cuda(non_blocking = True),
           'labels': data["labels"].cuda(non_blocking = True),
           'loss_mask': data["loss_mask"].cuda(non_blocking = True),
           'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
           'position_ids': data["position_ids"].cuda(non_blocking = True),
           'ref_seq':data['ref_seq']
       }
    #print("rank(step2):",mpu.get_tensor_model_parallel_rank(), batch.values())
    return batch.values()



def forward_step(final_emb,ref_seq,anno,table, model: GPTModel, score_mode=False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    
    prompt_len=len(final_emb)
      
        
    emb_tensor=torch.LongTensor(final_emb).unsqueeze(0)
    tokens= emb_tensor.cuda(non_blocking = True)
    attention_mask, loss_mask, position_ids=get_masks_and_position_ids(emb_tensor,prompt_len)
    attention_mask=attention_mask.cuda(non_blocking = True)
    position_ids=position_ids.cuda(non_blocking = True)
        
    # Get the batch.
    
        
    output_tensor = model(tokens, position_ids, attention_mask)
        #print(model)
        #for name,param in model.named_parameters():
        #    print(name,param)
    print(output_tensor)
        # do not add labels so that no labels are used
    print(output_tensor.shape)
    if not(score_mode):
        ranked_mut_list=[]
        if args.version==1:
            sorted_ids=torch.topk(output_tensor,1000,dim=2,sorted=True).indices
            mutation_pred=sorted_ids[:,-1]
            nuc_dic={0:'A',1:"T",2:"C",3:'G',4:'-'}
            
            for item in mutation_pred[0]:
                pos=(item-1)//5+1
                if pos>=1 and pos<=29903:
                    target_mut=nuc_dic[int((item-1).item())%5]
                    
                    mut=get_mutation(ref_seq,pos.item(),target_mut,anno,table)
                    if mut is not None:
                        #print(mut)
                        ranked_mut_list.append(copy.deepcopy(mut))
        return ranked_mut_list
    if score_mode:
        # shape: [1,length, muts]
        scored_tokens=tokens[1:]
        
        scoring=torch.gather(output_tensor,tokens)
            
    return output_tensor

     

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

def inference_mutgpt(forward_step_func,
             data_iterator,
             model_provider,
             verbose=False):
    """Evaluation."""
    initialize_megatron(extra_args_provider=None,args_defaults={})
    #args = parse_args(None, False)
    #print(args)
    variant_mutation_dic={}
    anno=read_designation(variant_mutation_dic)
    country_list=read_country_list()
    trans_table=read_table()
    args = get_args()
    timers = get_timers()
    ref_seq=read_ref()
    model_type=ModelType.encoder_or_decoder
   
    model= read_model(
        model_provider, model_type)
    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    reversion_allowed_pos=[670,22995,27810,29409]
    # make validation batch size independent from training batch size
    @app.route('/query',methods=['POST'])
    def online_query():
        print("收到请求头:", request.headers)
        print("原始请求体:", request.get_data())
        items = json.loads(request.get_data())
        base_variant=items['base_variant'].upper()
        if not(base_variant in variant_mutation_dic):
            return ["base variant invalid"]
        for item in items:
            print(item,items[item])
        additional_mutations=None
        if 'additional_mutations' in items:
            additional_mutations=items['additional_mutations']
        location=None
        date=None
        if 'location' in items:
            if items['location']!=None:
                location=items['location'].lower().replace(' ','').replace("'",'')
                if not(location in country_list):
                    location=None

        if 'date' in items:
            date=items['date']
        required_gene=None
        if 'required_gene' in items:
            required_gene=items['required_gene']
            required_gene=required_gene.upper()
            if len(required_gene)>=5:
                required_gene=required_gene[:-1]+required_gene[-1:].lower()
            
        max_count=10
        if 'max_count' in items:
            if items['max_count']!=None:
                try:
                    max_count=int(items['max_count'])
                except:
                    return ["max count incorrect"]
        location_time_embedding=location_time_encoding(location,date,country_list,version=1)
        additional_mutation_list=[]
        if additional_mutations is not None and len(additional_mutations)>1:
            additional_mutation_list=additional_mutations.split(',')
            for item in additional_mutation_list:
                try:
                    pos=int(item.strip()[1:-1])
                    if pos<=0 or pos>29903:
                        return ["Additional mutation invalid(pos out of range)"]
                    target=item.strip()[-1]
                    if not(target in ['A','T','C','G','-']):
                        return ["Additional mutation invalid(target nuc invalid)"]

                except:
                    return ["Additional mutation invalid"]
                
        print(additional_mutation_list)
        all_mutations=variant_mutation_dic[base_variant]
        print(all_mutations)
        all_mutations.extend(additional_mutation_list)
        mutation_embedding,final_seq=build_mutation_embedding(ref_seq,all_mutations,anno,trans_table,version=1)
        final_emb=location_time_embedding+mutation_embedding

        

        with torch.no_grad():
            returned_mutlist = forward_step(final_emb,final_seq,anno,trans_table,model=model)

        output_mutlist=[]
    
        for i in range(len(returned_mutlist)):
            item=returned_mutlist[i]
            accept=True
            if required_gene is not None:
                accept=False
                for mut in item:
                    if required_gene+':' in mut:
                        accept=True
            if accept:
                pos=int(item[0][1:-1])
                #revert to original without being a well-known reversive site, likely influenced by artefact
                if not(pos in reversion_allowed_pos) and (item[0][-1]==ref_seq[pos-1]):
                    accept=False
            
            if accept:
                if len(output_mutlist)<max_count:
                    output_mutlist.append(copy.deepcopy(item))
                    for j in range(len(output_mutlist[-1])):
                        if output_mutlist[-1][j][-1]=='O':
                            output_mutlist[-1][j]=output_mutlist[-1][j][:-1]+'stop'

        if args.empty_unused_memory_level >= 1:
            torch.cuda.empty_cache() 
        return output_mutlist

    @app.route('/score',methods=['POST'])
    def online_score():
        print("收到请求头:", request.headers)
        print("原始请求体:", request.get_data())
        items = json.loads(request.get_data())
        base_variant=items['base_variant'].upper()
        if not(base_variant in variant_mutation_dic):
            return ["base variant invalid"]
        for item in items:
            print(item,items[item])
        additional_mutations=None
        if 'additional_mutations' in items:
            additional_mutations=items['additional_mutations']
        location=None
        date=None
        if 'location' in items:
            if items['location']!=None:
                location=items['location'].lower().replace(' ','').replace("'",'')
                if not(location in country_list):
                    location=None

        if 'date' in items:
            date=items['date']
        
        location_time_embedding=location_time_encoding(location,date,country_list,version=1)
        additional_mutation_list=[]
        if additional_mutations is not None and len(additional_mutations)>1:
            additional_mutation_list=additional_mutations.split(',')
            for item in additional_mutation_list:
                try:
                    pos=int(item.strip()[1:-1])
                    if pos<=0 or pos>29903:
                        return ["Additional mutation invalid(pos out of range)"]
                    target=item.strip()[-1]
                    if not(target in ['A','T','C','G','-']):
                        return ["Additional mutation invalid(target nuc invalid)"]

                except:
                    return ["Additional mutation invalid"]
                
        print(additional_mutation_list)
        all_mutations=variant_mutation_dic[base_variant]
        print(all_mutations)
        all_mutations.extend(additional_mutation_list)
        mutation_embedding,final_seq=build_mutation_embedding(ref_seq,all_mutations,anno,trans_table,version=1)
        final_emb=location_time_embedding+mutation_embedding

        

        with torch.no_grad():
            returned_mutlist = forward_step(final_emb,final_seq,anno,trans_table,model=model)

       
        return output_mutlist
    return 0


def build_mutation_embedding(ref_seq,mutation_list,anno,trans_table,version=1):
    current_seq=copy.deepcopy(ref_seq)
    current_encoding=[]
    #print(mutation_list)

    for i in range(len(mutation_list)):
        mutation=mutation_list[i]
        if len(mutation)>2:
            encoding,new_seq=mutation_encoding(current_seq,mutation,anno,trans_table,version=1)
            
            if encoding is not None:
            
                current_encoding.extend(copy.deepcopy(encoding))
                current_seq=copy.deepcopy(new_seq)

    return current_encoding,current_seq


def input_iterator(path,version=0):
    f=open(path,'r')
    lines=f.readlines()
    f.close()
    variant_mutation_dic={}
    anno=read_designation(variant_mutation_dic)
    country_list=read_country_list()
    ref_seq=read_ref()
    trans_table=read_table()

    for l in lines:
        print(l)
        place,date,base_variant,additional_mutations=l.split()
        additional_mutation_list=[]
        if additional_mutations!='-':
            additional_mutation_list=additional_mutations.split(',')
        all_mutations=variant_mutation_dic[base_variant]
        all_mutations.extend(additional_mutation_list)
        mutation_embedding,final_seq=build_mutation_embedding(ref_seq,all_mutations,anno,trans_table,version=version)
        location_time_embedding=location_time_encoding(place.lower(),date,country_list,version=version)
        final_emb=location_time_embedding+mutation_embedding
        prompt_len=len(final_emb)
        
        emb_tensor=torch.LongTensor(final_emb).unsqueeze(0)
        attention_mask, loss_mask, position_ids=get_masks_and_position_ids(emb_tensor,prompt_len)
        dat={}
        dat['tokens']=emb_tensor
        dat['attention_mask']=attention_mask
        dat['loss_mask']=loss_mask
        dat['position_ids']=position_ids
        dat['labels']=emb_tensor
        dat['ref_seq']=final_seq
        yield dat

    return 0





if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True
    
    path="../queries.txt"
    
    data_iterator=input_iterator(path,version=1)

    inference_mutgpt(forward_step,
        data_iterator,
        model_provider
    )
    app.run(host="0.0.0.0", port=5001, threaded=False)
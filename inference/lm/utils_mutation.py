# written by Xu Zou
# necessary changes for functions to process GPT-for-mutation. 

# megatron/training/utils.py
import os
import random
import copy
import numpy as np
import torch
import json
from megatron.training import print_rank_0
def read_variant_table(path):
    w=open(path,'r')
    lines=w.readlines()
    w.close()
    variant_table={}
    for line in lines:
        l=json.loads(line)
        variant_table[l['variant']]=l['encoding']
    return variant_table
def get_masks_and_position_ids(data,prompt_len):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    
    micro_batch_size, seq_length = data.size()
    

    
    # Attention mask (lower triangular).
   
    att_mask_batch = micro_batch_size

    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length),device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # attention mask for mutation embedding (each mutation is encoded to 12 tokens, these 9 tokens form 1-2-3-3 groups that do not link with each other to avoid leakage)
    '''
    for q in range((seq_length-5)//12):
        attention_mask[:,:,12*q+9:12*q+17,12*q+8]=0
        attention_mask[:,:,12*q+11:12*q+17,12*q+9:12*q+11]=0
        attention_mask[:,:,12*q+14:12*q+17,12*q+11:12*q+14]=0
    '''
    #print_rank_0("set up specific att mask")
    #print_rank_0(attention_mask[:,:,-12:,-12:])
    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float,device=data.device)
    loss_mask[:,:prompt_len-1]=0

    #loss mask is applied on labels, so the prompt_len-1 shall have loss
    
    #loss_mask[:,-1]=0
    # Position ids.
    
    position_ids = torch.arange(seq_length, dtype=torch.long,device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids

# embedding transform, 9-dim version to 12-dim version

def embedding_transform(prompt,version=0,shuffle=False):
    new_prompt=[]
    labels=[]
    if version==1:
        prt=copy.deepcopy(prompt)
        if shuffle:
            random.shuffle(prt)
        
        return prt,prt
    if version==0:
        for i in range(len(prompt)//9):
            new_prompt.extend([0,0,0])
            new_prompt.extend(copy.deepcopy(prompt[i*9:i*9+9]))
            labels.extend(
            [prompt[i*9],prompt[i*9+1],prompt[i*9+3],prompt[i*9+6],0,prompt[i*9+2],0,prompt[i*9+4],prompt[i*9+5],0,prompt[i*9+7],prompt[i*9+8]])
            
        return new_prompt,labels
    if version>=2:
        version_length={2:4,3:7}
        l=version_length[version]
        
        for i in range(len(prompt)//l):
            for j in range(l):
                new_prompt.append(0)
            new_prompt.extend(copy.deepcopy(prompt[i*l:i*l+l]))
            labels.extend(copy.deepcopy(prompt[i*l:i*l+l]))
            for j in range(l):
                labels.append(0)
        return new_prompt,labels
    
def get_step_data_random(dir,num_samples,variants_encoding_dict,version=0):
    # randomly choosing training data is not worse than epochs
    files=os.listdir(dir)
    is_chosen=False
    while not(is_chosen):
        fname=random.choice(files)
        if '.json' in fname:
            is_chosen=True
    f=open(dir+'/'+fname,'r')
    l=f.readlines()
    batch_data=random.sample(l,num_samples)
    # pad it to be 0 0 0 
    max_variant_encoding_length=0
    max_sequence_encoding_length=0
    all_encodings=[]
    
    lc_encoding=dat['location_time_encoding']
    variant_encoding=variants_encoding_dict[dat['variant']]
        
    sequence_encoding=dat['sequence_encoding']
    
    data=[]
    data=copy.deepcopy(lc_encoding)
    var,var_labels=embedding_transform(variant_encoding,version=version)
    
    seq,seq_labels=embedding_transform(sequence_encoding,version=version)
    data.extend(var)
    prompt_len=len(data)
    data.extend(seq)
    data_len=len(data)
    
    labels=[0,0,0,0]
    if version>=2:
        labels.append(0)
    labels.extend(var_labels)
    labels.extend(seq_labels)
   
    zero_mask=np.ones([1,data_len])
    for i in range(data_len):
        if labels[i]==0:
            zero_mask[0,i]=0
    if version>=2:
        labels=labels[:-1]

    data=torch.LongTensor(data).view(1,data_len)
    labels=torch.LongTensor(labels).view(1,data_len-1)
    attention_mask,loss_mask,position_ids=get_masks_and_position_ids(torch.LongTensor(data),prompt_len)
    loss_mask=loss_mask*zero_mask
    return data,attention_mask,loss_mask,position_ids,labels

# data iterator, simply goes through all data.
def data_process(batched_data,variants_encoding_dict,version=0,shuffle=False):
    max_variant_encoding_length=0
    max_sequence_encoding_length=0
    all_encodings=[]
    label_encodings=[]
    num_samples=len(batched_data)
    has_weight=0
    for datpt in batched_data:
        dat=json.loads(datpt)
        lc_encoding=dat['location_time_encoding']
        if 'weight' in dat:
            has_weight=1
            weight_encoding=dat['weight']
        variant_encoding=variants_encoding_dict[dat['variant']]
        variant_encoding,variant_label=embedding_transform(variant_encoding,version=version,shuffle=shuffle)
        if len(variant_encoding)>max_variant_encoding_length:
            max_variant_encoding_length=len(variant_encoding)
            #print(dat['variant'],max_variant_encoding_length,variant_encoding)
        sequence_encoding=dat['sequence_encoding']
        sequence_encoding,sequence_label=embedding_transform(sequence_encoding,version=version)
        if len(sequence_encoding)>max_sequence_encoding_length:
            max_sequence_encoding_length=len(sequence_encoding)
            #print(dat['sequence_encoding'],max_sequence_encoding_length)
        all_encodings.append([copy.deepcopy(lc_encoding),copy.deepcopy(variant_encoding),copy.deepcopy(sequence_encoding)])
        label_encodings.append([copy.deepcopy(variant_label),copy.deepcopy(sequence_label)])

    prompt_len=max_variant_encoding_length+5
    seq_len=max_sequence_encoding_length
    data=np.zeros([num_samples,prompt_len+seq_len],dtype=int)
    labels=np.zeros([num_samples,prompt_len+seq_len],dtype=int)
    # padding
    for i in range(len(all_encodings)):
        data[i,0:5]=all_encodings[i][0]
        l1=len(all_encodings[i][1])
        data[i,prompt_len-l1:prompt_len]=all_encodings[i][1]
        if version<2:
            labels[i,prompt_len-l1-1:prompt_len-1]=label_encodings[i][0]
        else:
            labels[i,prompt_len-l1:prompt_len]=label_encodings[i][0]
        l2=len(all_encodings[i][2])
        data[i,prompt_len:prompt_len+l2]=all_encodings[i][2]
        if version<2:
            labels[i,prompt_len-1:prompt_len+l2-1]=label_encodings[i][1]
        else:
            labels[i,prompt_len:prompt_len+l2]=label_encodings[i][1]

    zero_mask=np.ones([1,prompt_len+seq_len])
    for i in range(prompt_len+seq_len):
        if labels[0,i]==0:
            zero_mask[0,i]=0
    
    
    attention_mask,loss_mask,position_ids=get_masks_and_position_ids(torch.LongTensor(data),prompt_len)
    loss_mask=loss_mask*zero_mask

    dat={}
    dat['tokens']=torch.LongTensor(data[:,:-1])
    dat['attention_mask']=attention_mask[:,:,:-1,:-1]
    dat['loss_mask']=loss_mask[:,:-1]
    dat['position_ids']=position_ids[:,:-1]
    
    dat['labels']=torch.LongTensor(labels[:,:-1])
    if has_weight==1:
        dat['weight']=weight_encoding

    
    return dat

def split_filelist(dir, train_ratio, valid_ratio, test_ratio):
    files=os.listdir(dir)
    all_list=[]
    for fname in files:
        if '.json' in fname:
            all_list.append(fname)
    random.shuffle(all_list)

    train_num=int(train_ratio*len(all_list))
    valid_num=int(valid_ratio*len(all_list))
    test_num=int(test_ratio*len(all_list))
    train_remain=train_ratio*len(all_list)-train_num
    valid_remain=valid_ratio*len(all_list)-valid_num
    test_remain=test_ratio*len(all_list)-test_num

    while train_num+valid_num+test_num<len(all_list):
        if train_remain>valid_remain and train_remain>test_remain:
            train_remain-=1
            train_num+=1
        else:
            if valid_remain>test_remain:
                valid_remain-=1
                valid_num+=1
            else:
                test_remain-=1
                test_num+=1
    train_list=all_list[:train_num]
    valid_list=all_list[train_num:train_num+valid_num]
    test_list=all_list[train_num+valid_num:]
    return train_list,valid_list,test_list


# 184300-184310   time(2019-2024)   dimension 2
# 184311-184382   time (2019.1--2024.12) dimension 3 
# 184401-184531   date dimension 4
def get_date_factor(date_end,tensor,date_decay_value=0.1,version=0):
    if date_end is None:
        return 1
    tensor=json.loads(tensor)['location_time_encoding']
    version_base_dic={0:183921,1:149600,2:190600, 3:192100}
    vbase=version_base_dic[version]
    tensor_month=tensor[3]-vbase-390     #1 refers to 201901
    tensor_date=tensor[4]-vbase-480
    end_yy,end_mm,end_dd=date_end.split('-')
    end_month=(int(end_yy)-2019)*12+int(end_mm)
    decay_base=end_month-tensor_month
    return np.exp(-decay_base*date_decay_value)

def compare_date(date_range,tensor,version=0):
    #print(date_range,tensor)
    tensor=json.loads(tensor)['location_time_encoding']
    if date_range is None:
        return True
    # if there is no date range limit, always pass
    date_start,date_end=date_range
    version_base_dic={0:183921,1:149600,2:190600, 3:192100}
    vbase=version_base_dic[version]
    tensor_month=tensor[3]-vbase-390     #1 refers to 201901
    tensor_date=tensor[4]-vbase-480
    start_yy,start_mm,start_dd=date_start.split('-')
    end_yy,end_mm,end_dd=date_end.split('-')
    start_month=(int(start_yy)-2019)*12+int(start_mm)
    end_month=(int(end_yy)-2019)*12+int(end_mm)
    start_date=int(start_dd)
    end_date=int(end_dd)

    if start_month>tensor_month:
        return False
    if end_month<tensor_month:
        return False
    if start_month==tensor_month and start_date>tensor_date:
        return False
    if end_month==tensor_month and end_date<tensor_date:
        return False
    #print(date_range,tensor)
    
    #print("accepted")
    return True

    
def data_iterator_sample(dir,variants_encoding_dict,file_list=None,date_range=None,total_ranks=1,process=True,
                         country_id=None,version=0,evaluation=False):
    print("building dataloader")
    files=os.listdir(dir)
    # compute date_range-related tensor
    #random.shuffle(files)
    #print(files[0:100])
    if file_list is not None:
        files=file_list
    ff=[]
    for item in files:
        if '.json'==item[-5:]:
            ff.append(item)
    ff=sorted(ff)
    current_pos=0
    num_samples=1
    if total_ranks>1:
        global_rank=torch.distributed.get_rank()
        print("rank is:",global_rank)
    else:
        global_rank=0
    total_pos=0
    current_samples=0
    #print(dir,files[0:10])
    current_epoch=1
    while 1:
        '''
        
        if global_rank==0:
            random.shuffle(files)
        torch.distributed.barrier()
        torch.broadcast(files,0)
        
        '''
        for i in range(len(ff)):
            fname=ff[i]

            #print(fname)
            f=open(dir+'/'+fname,'r')
                
            lines=f.readlines()
                
            f.close()
            while len(lines[-1])<2: 
                lines=lines[:-1]
                #random.shuffle(lines)
            current_pos=0
                
                #num_samples= yield
            while current_pos<len(lines):
                data=lines[current_pos:current_pos+1]
                
                if compare_date(date_range,data[0],version=version):
                    lcc=json.loads(data[0])['location_time_encoding'][0]
                    if (country_id is None) or lcc==country_id:
                        
                        if total_pos%total_ranks==global_rank:
                            
                            if process:
                                if not(evaluation) or len(json.loads(data[0])['sequence_encoding'])>0:

                                    output_data=data_process(data,variants_encoding_dict,version=version)
                                    #print(output_data)
                                    
                                    yield output_data
                                    
                            else:
                                if not(evaluation) or len(json.loads(data[0])['sequence_encoding'])>0:
                                    yield data
                        total_pos+=1
                        
                    #print("output data yielded")
                current_pos+=1
                    
                    #num_samples= yield

        if evaluation:
            while 1:
                yield None
        current_epoch+=1
        print("current epoch:",current_epoch)
            # in evaluation mode, the data iterator stops providing data after 1 epoch.
                
    output_data=data_process(current_data,variants_encoding_dict)
    if compare_date(date_range,data[0],version=version):
        yield output_data

def data_iterator_weighted(dir,variants_encoding_dict,file_list=None,date_end=None,total_ranks=1,process=True,
                         country_id=None,version=0,evaluation=False,shuffle_muts=False):
    print("building dataloader")
    files=os.listdir(dir)
    # compute date_range-related tensor
    #random.shuffle(files)
    #print(files[0:100])
    if file_list is not None:
        files=file_list
    
    current_pos=0
    num_samples=1
    if total_ranks>1:
        global_rank=torch.distributed.get_rank()
        #print("rank is:",global_rank)
    else:
        global_rank=0
    total_pos=0
    current_samples=0
    #print(dir,files[0:10])
    current_epoch=1
    sumweight=0
    threshold=10  # sum of everything is ~200 billion, if we use year decay (exp(-month*1/30)? likely result in 60 billion or so. 
    # step size=256 so to browse through the whole dataset it takes 200 million for each task 200 million/5000=40k. 
    while 1:
        random.shuffle(files)
       
        for i in range(len(files)//total_ranks):
            fname=files[i*total_ranks+global_rank]
            if ('.json' in fname) :
                
                f=open(dir+'/'+fname,'r')
                lines=f.readlines()
                f.close()
                while len(lines[-1])<2: 
                    lines=lines[:-1]
                #random.shuffle(lines)
                current_pos=0
                
                #num_samples= yield
                random.shuffle(lines)
                while current_pos<len(lines):
                    data=lines[current_pos:current_pos+1]
                    
                   
                    lcc=json.loads(data[0])['location_time_encoding'][0]
                    date_factor=get_date_factor(date_end,data[0],version=version)
                    wt=np.log(json.loads(data[0])['weight']/100)+1
                
                    sumweight+=wt*date_factor
                    if sumweight>threshold:
                        while sumweight>threshold:
                            sumweight-=threshold

                        if process:
                                    
                            output_data=data_process(data,variants_encoding_dict,version=version,shuffle=shuffle_muts)
                                    #print(output_data)
                            yield output_data
                                    
                        else:
                            yield data
                    total_pos+=1
                          
                    #print("output data yielded")
                    current_pos+=1
                    
                    #num_samples= yield

                current_data=lines[current_pos:]
                current_samples=len(current_data)
        if evaluation:
            while 1:
                yield None
        current_epoch+=1
        print("current epoch:",current_epoch)
            # in evaluation mode, the data iterator stops providing data after 1 epoch.
                
    output_data=data_process(current_data,variants_encoding_dict)
    if compare_date(date_range,data[0],version=version):
        yield output_data
'''
from megatron.core.datasets.megatron_dataset import MegatronDataset
from typing import Dict, Optional, Tuple

class MutDataset:
    #Mutation dataset formatted as megatron dataset

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        try:
            self._pad_token_id = self.config.tokenizer.pad
        except:
            self._pad_token_id = _PAD_TOKEN_ID

        (
            self.document_index,
            self.sample_index,
            self.shuffle_index,
        ) = self._build_document_sample_shuffle_indices()

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        """Abstract method implementation

        Returns:
            int: The length of the dataset
        """
        return self.sample_index.shape[0] - 1

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Abstract method implementation

        Args:
            idx (Optioal[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # For padded sequences, mask the loss
        loss_mask[labels == self._pad_token_id] = 0.0

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    def _query_document_sample_shuffle_indices(
        self, idx: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Get the text (token ids) and document ids for a given index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The text ids and document ids
        """
        # Do the shuffle mapping
        idx = self.shuffle_index[idx]

        # Get the beginning and end documents and offsets
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        document_ids = []
        sample_parts = []

        # Sample spans a single document
        if doc_index_beg == doc_index_end:
            # Add the document id
            document_ids.append(self.document_index[doc_index_beg])

            # Add the entire sample
            sample_parts.append(
                self.dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )

        # Sample spans multiple documents
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                # Add the document id
                document_ids.append(self.document_index[i])

                # Add the sample part
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                sample_parts.append(
                    self.dataset.get(self.document_index[i], offset=offset, length=length)
                )
        assert len(document_ids) == len(
            sample_parts
        ), f"len(document_ids) ({len(document_ids)}) != len(sample_parts) ({len(sample_parts)})"

        length = sum(map(len, sample_parts))

        # Pad the sample if necessary
        if length < (self.config.sequence_length + self.config.add_extra_token_to_sequence):
            sample_parts.append(
                [self._pad_token_id]
                * (self.config.sequence_length + self.config.add_extra_token_to_sequence - length)
            )

        return (
            numpy.concatenate(sample_parts, dtype=numpy.int64),
            numpy.array(document_ids, dtype=numpy.int64),
        )

    def _build_document_sample_shuffle_indices(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Build the document index, the sample index, and the shuffle index
        
        The document index:
            -- 1-D
            -- An ordered array of document ids

        The sample index:
            -- 2-D
            -- The document indices and offsets which mark the start of every sample

        The shuffle index:
            -- 1-D
            -- A random permutation of index range of the sample index

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The document index, the sample index, and the shuffle index
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            get_path_to = lambda suffix: os.path.join(
                path_to_cache,
                f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}-{suffix}",
            )
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_document_index,
                        path_to_sample_index,
                        path_to_shuffle_index,
                    ],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):

            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )
            t_beg = time.time()

            sequence_length = self.config.sequence_length
            num_tokens_per_epoch = self._get_num_tokens_per_epoch()
            num_epochs = self._get_num_epochs(num_tokens_per_epoch)

            if num_epochs == 1:
                separate_final_epoch = False
            else:
                # Get the number of samples for the last epoch
                num_samples_sans_final_epoch = (
                    (num_epochs - 1) * num_tokens_per_epoch
                    - self.config.add_extra_token_to_sequence
                ) // sequence_length
                num_samples_from_final_epoch = self.num_samples - num_samples_sans_final_epoch
                num_samples_per_epoch = (
                    num_tokens_per_epoch - self.config.add_extra_token_to_sequence
                ) // sequence_length

                # num_samples_from_final_epoch should be non-negative
                assert num_samples_from_final_epoch >= 0

                # num_samples_from_final_epoch should not exceed max value
                assert num_samples_from_final_epoch <= num_samples_per_epoch + 1

                # Separate the final epoch if it falls below the threshold
                threshold = 0.80
                separate_final_epoch = num_samples_from_final_epoch < int(
                    threshold * num_samples_per_epoch
                )

                log_single_rank(
                    logger,
                    logging.DEBUG,
                    f"> num_samples_from_final_epoch: {num_samples_from_final_epoch}",
                )
                log_single_rank(logger, logging.DEBUG, f"> threshold: {threshold}")
                log_single_rank(
                    logger, logging.DEBUG, f"> num_samples_per_epoch: {num_samples_per_epoch}"
                )

            log_single_rank(
                logger, logging.DEBUG, f"> separate_final_epoch: {separate_final_epoch}"
            )

            numpy_random_state = numpy.random.RandomState(self.config.random_seed)

            # Build the document index
            document_index = _build_document_index(
                self.indices, num_epochs, numpy_random_state, separate_final_epoch
            )

            drop_last_partial_sequence = True
            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence

            # Build the sample index
            from megatron.core.datasets import helpers

            if self.index_split == Split.valid:
                drop_last_partial_sequence = self.config.drop_last_partial_validation_sequence
            else:
                drop_last_partial_sequence = True

            assert document_index.dtype == numpy.int32
            assert self.dataset.sequence_lengths.dtype == numpy.int32
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                # Heuristic: if "access density" of sequence_lengths is relatively high,
                # force loading the mmap-ed array into memory by taking a copy.
                # System performance benefits come from two aspects:
                # 1. **sequentially** pre-loading the whole file if we're gonna read a large fraction anyways.
                # 2. GIL is held when calling into c++ code; making the c++ func faster improves parallelism.
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths
            sample_index = helpers.build_sample_idx(
                sequence_lengths_for_cpp,
                document_index,
                sequence_length,
                num_epochs,
                num_tokens_per_epoch,
                drop_last_partial_sequence,
                self.config.add_extra_token_to_sequence,
            )

            # Build the shuffle index
            if separate_final_epoch:
                shuffle_index = _build_shuffle_index(
                    num_samples_sans_final_epoch, sample_index.shape[0] - 1, numpy_random_state
                )
            else:
                shuffle_index = _build_shuffle_index(
                    sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
                )

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                # Write the description
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                numpy.save(path_to_document_index, document_index, allow_pickle=True)
                numpy.save(path_to_sample_index, sample_index, allow_pickle=True)
                numpy.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save the {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            log_single_rank(
                logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
            )
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index, sample_index, shuffle_index

        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the document index from {os.path.basename(path_to_document_index)}",
        )
        t_beg = time.time()
        document_index = numpy.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}",
        )
        t_beg = time.time()
        sample_index = numpy.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger,
            logging.INFO,
            f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}",
        )
        t_beg = time.time()
        shuffle_index = numpy.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(
            logger, logging.INFO, f"> total number of samples: {sample_index.shape[0] - 1}"
        )

        return document_index, sample_index, shuffle_index

    def _get_num_tokens_per_epoch(self) -> int:
        """Calculate the number of tokens in a single epoch

        Returns:
            int: The number of tokens in a single epoch
        """
        return int(numpy.sum(self.dataset.sequence_lengths[self.indices]))

    def _get_num_epochs(self, num_tokens_per_epoch: int) -> int:
        """Calculate the number of epochs

        Args:
            num_tokens_per_epoch (int): The number of tokens in a single epoch

        Returns:
            int: The number of epochs
        """
        num_epochs = 1
        num_tokens = num_tokens_per_epoch
        if self.num_samples is None:
            return num_epochs
        else:
            num_tokens_requested = (
                self.num_samples * self.config.sequence_length
            ) + self.config.add_extra_token_to_sequence
            while num_tokens < num_tokens_requested:
                num_epochs += 1
                num_tokens += num_tokens_per_epoch
        return num_epochs
'''

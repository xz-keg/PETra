


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
def get_masks_and_position_ids(data,prompt_len,seq_lengths=None):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    
    micro_batch_size, seq_length = data.size()
    
    if micro_batch_size==1:
    
    # Attention mask (lower triangular).
   
        att_mask_batch = micro_batch_size
        #print(data,prompt_len)
        try:
            prompt_len=prompt_len[0]
        except:
            prompt_len=prompt_len

        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length),device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)

        loss_mask = torch.ones(data.size(), dtype=torch.float,device=data.device)
        loss_mask[:,:prompt_len-1]=0
        
        position_ids = torch.arange(seq_length, dtype=torch.long,device=data.device)
        position_ids = position_ids.unsqueeze(0).expand_as(data)
    
        attention_mask = (attention_mask < 0.5)

        return attention_mask, loss_mask, position_ids
    else:
        attention_mask = torch.zeros((micro_batch_size, seq_length, seq_length), device=data.device)
        
        # 初始化 loss_mask 和 position_ids
        loss_mask = torch.zeros((micro_batch_size, seq_length), dtype=torch.float, device=data.device)
        position_ids = torch.zeros((micro_batch_size, seq_length), dtype=torch.long, device=data.device)
        
        # 为每个样本单独处理
        for i in range(micro_batch_size):
            actual_len = seq_lengths[i]  # 当前样本实际长度
            p_len = prompt_len[i]        # 当前样本提示长度
            
            # 1. 构建 attention_mask
            #    有效区域（actual_len x actual_len）设置为下三角矩阵（值为1）
            attention_mask[i, :actual_len, :actual_len] = torch.tril(
                torch.ones(actual_len, actual_len, device=data.device))
            
            # 2. 构建 loss_mask
            #    只有生成部分（prompt之后的有效token）设为1
            loss_mask[i, p_len:actual_len] = 1
            
            # 3. 构建 position_ids
            #    有效位置填充0到actual_len-1的位置ID
            position_ids[i, :actual_len] = torch.arange(actual_len, device=data.device)
        
        # 转换 attention_mask: 
        #   - 原始矩阵中0表示需要mask的位置，1表示保留的位置
        #   - 转换为布尔矩阵：<0.5 的位置变为True（需要mask），其余为False
        attention_mask = (attention_mask < 0.5)
        
        # 调整attention_mask维度: [batch, 1, seq, seq]
        attention_mask = attention_mask.view(micro_batch_size, 1, seq_length, seq_length)
        
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

def data_process_multiple(batched_data,variants_encoding_dict,version=0,shuffle=False):
    max_variant_encoding_length=0
    max_sequence_encoding_length=0
    all_encodings=[]
    label_encodings=[]
    prompt_lens=[]
    seq_lens=[]
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
        prompt_lens.append(len(variant_encoding)+5)
        seq_lens.append(len(sequence_encoding)+prompt_lens[-1])
        
            #print(dat['sequence_encoding'],max_sequence_encoding_length)
        all_encodings.append([copy.deepcopy(lc_encoding),copy.deepcopy(variant_encoding),copy.deepcopy(sequence_encoding)])
        label_encodings.append([copy.deepcopy(variant_label),copy.deepcopy(sequence_label)])
    max_seq_len=np.max(seq_lens)
    
    data=np.zeros([num_samples,max_seq_len],dtype=int)
    labels=np.zeros([num_samples,max_seq_len],dtype=int)
    # padding
    for i in range(len(all_encodings)):
        data[i,0:5]=all_encodings[i][0]
        l1=len(all_encodings[i][1])
        prompt_len=prompt_lens[i]
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

    
    
    attention_mask,loss_mask,position_ids=get_masks_and_position_ids(torch.LongTensor(data),prompt_lens,seq_lengths=seq_lens)
    #loss_mask=loss_mask*zero_mask

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


# 184300-184310   time(2019-2025)   dimension 2
# 184311-184394   time (2019.1--2025.12) dimension 3 
# 184401-184531   date dimension 4
def get_date_factor(date_end,tensor,date_decay_value=0.06,version=0):
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
                         country_id=None,version=0,evaluation=False,mbatch_size=1):
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
    current_data=[]
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
                            current_data.append(copy.deepcopy(data[0]))
                            if len(current_data)>=mbatch_size:
                                if process:
                                    if not(evaluation) or len(json.loads(data[0])['sequence_encoding'])>0:

                                        output_data=data_process_multiple(current_data,variants_encoding_dict,version=version)
                                        #print(output_data)
                                        
                                        yield output_data
                                        
                                else:
                                    if not(evaluation) or len(json.loads(data[0])['sequence_encoding'])>0:
                                        yield current_data
                                current_data=[]

                                
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
                
    output_data=data_process_multiple(current_data,variants_encoding_dict,version=version)
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
            if ('.json'==fname[-5:]) :
                
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

def data_iterator_multi(dir,variants_encoding_dict,file_list=None,date_end=None,total_ranks=1,process=True,
                         country_id=None,version=1,evaluation=False,shuffle_muts=False,weighted=0.08,mbatch_size=4,use_weighted=True):
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
        current_data=[]
        
        for i in range(len(files)//total_ranks):
            fname=files[i*total_ranks+global_rank]
            if ('.json'==fname[-5:]) :
                
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
                    date_factor=get_date_factor(date_end,data[0],version=version,date_decay_value=weighted)
                    wt=np.log(json.loads(data[0])['weight']/100)+1
                    if not use_weighted:
                        wt=15
                    #final_weight=wt*date_factor
                    sumweight+=wt*date_factor
                    #print(wt, date_factor,use_weighted)
                    if not use_weighted and abs(weighted)<0.0001:
                        sumweight=11 # force select all data without looking at weight.
                    if sumweight>threshold:
                        while sumweight>threshold:
                            sumweight-=threshold
                        current_data.append(copy.deepcopy(data[0]))
                        if len(current_data)>=mbatch_size:

                            if process:
                                        
                                output_data=data_process_multiple(current_data,variants_encoding_dict,version=version,shuffle=shuffle_muts)
                                        #print(output_data)
                                yield output_data
                                
                                        
                            else:
                                yield current_data
                            current_data=[]

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
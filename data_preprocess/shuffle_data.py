import os
import random
import json
import numpy as np
import copy
all_seqs=[]
import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--date', type=str,default='2025-07-16')
parser.add_argument('--data-dir', type=str,default=None)
args = parser.parse_args()
data_dir=args.data_dir
date=args.date


def shuffle_data(path,weight_func,version=0,dir_name="shuffled_seqs_weight",keep_weight=False):
    print("process data with version ",version)
    flist=os.listdir(path)
    if version>0:
        dir_text='seqs_v'+str(version)
    else:
        dir_text='seqs'
    if not(dir_text in flist):
        return 0
    
    filelist=os.listdir(path+'/'+dir_text)
    all_seqs=[]

    sum_weight=0
    num_data=0
    for item in filelist:
        if '.json' in item:
            f=open(path+'/'+dir_text+'/'+item,'r')
            lines=f.readlines()
            f.close()
            for l in lines:
                num_data+=1
                if num_data%1000000==0:
                    print("finished loading seqs: ",num_data)
                seq=json.loads(l)
                for ll in range(len(seq['sequence_encoding'])//9):
                    seq['sequence_encoding'][9*ll+4]=int(seq['sequence_encoding'][9*ll+4]+0.1)
                    seq['sequence_encoding'][9*ll+7]=int(seq['sequence_encoding'][9*ll+7]+0.1)
               
                wt=int(weight_func(seq['weight'])+0.5)
                if not(keep_weight):
                    del seq['weight']
                # this uses a simple method, it makes several copies of each seq according to their weight. 
                for i in range(wt):
                    all_seqs.append(copy.deepcopy(seq))
                
    random.shuffle(all_seqs)
    dir_out=dir_name
    if version>0:
        dir_out=dir_name+'_v'+str(version)
    if not(dir_out in flist):
        os.mkdir(path+'/'+dir_out)
    current_count=0
    current_weight=0
    f=open(path+'/'+dir_out+'/'+str(current_count)+'.json','w')
    current_items=0
    all_count=0
    for item in all_seqs:
        print(json.dumps(item),file=f)
        current_items+=1
        all_count+=1
        if current_items>len(all_seqs)*0.0002:
            f.close()
            current_count+=1
            current_items=0
            f=open(path+'/'+dir_out+'/'+str(current_count)+'.json','w')
            print('finished_processing files: ',current_count)
            print('finished_processing seqs: ',all_count)
    f.close()
    return 0

def trans(weight):
    return max(np.log(weight/100),1)

def uniform(weight):
    return 1

if data_dir is None:
    data_dir='../data'+date
dirs=os.listdir(data_dir)
for item in dirs:
    if item=='seqs': 
        w=shuffle_data(data_dir,uniform,dir_name="shuffled_seqs",keep_weight=True) 
    if 'seqs_v' in item:
        version=int(item[-1])
        w=shuffle_data(data_dir,uniform,version=version,dir_name="shuffled_seqs",keep_weight=True)


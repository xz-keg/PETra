from utils_mutation import *

data_dir_prefix=""
def data_count(data_dir,date_range,full_range):
    print("counting dataloader")
    files=os.listdir(data_dir)
   
    
    current_pos=0
    num_samples=1
    
    total_pos=0
    
    #print(dir,files[0:10])
    current_epoch=1
    total_samples=0
    current_samples=0
    available_samples=0
    inrange_samples=0

       
    
    for fname in files:
        if ('.json' in fname) :
                
            f=open(data_dir+'/'+fname,'r')
            lines=f.readlines()
            f.close()
            while len(lines[-1])<2: 
                lines=lines[:-1]
                #random.shuffle(lines)
            current_pos=0
                
                #num_samples= yield
            while current_pos<len(lines):
                data=lines[current_pos:current_pos+1]
                
                if compare_date(date_range,data[0],version=1):
                    inrange_samples+=1
                if compare_date(full_range,data[0],version=1):
                    available_samples+=1
                total_samples+=1
                if total_samples%1000000==0:
                    print('processed samples:',total_samples)
                    #print("output data yielded")
                current_pos+=1
                    
    print(date_range,total_samples,inrange_samples,available_samples)

    return total_samples,inrange_samples,available_samples
date1='2025-07-16'
date2='2025-02-12'
drange1=['2025-02-13',date1]
drange2=['2019-01-01',date1]
data_dir=data_dir_prefix+date1+'/seqs_v1'
total_samples,inrange_samples,available_samples=data_count(data_dir,drange1,drange2)

drange1=['2019-01-01',date2]
drange2=['2019-01-01',date2]
data_dir=data_dir_prefix+date2+'/seqs_v1'
total_samples,inrange_samples,available_samples=data_count(data_dir,drange1,drange2)

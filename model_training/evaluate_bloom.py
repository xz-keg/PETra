import json
import numpy as np
from utils import *
bloom_dir="bloom_1106"
data_dir_prefix=""
eval_clade='24A'  #only option as bloom estimator does not separate JN.1-descendants
eval_month=None
variant_mutation_dic={}
anno=read_designation(variant_mutation_dic)
trans_table=read_table()




def read_bloom(path,ref_dic):
    if eval_clade=='23I':
        ref_base=ref_dic['BA.2.86 0']
    if eval_clade=='24A':
        ref_base=ref_dic['JN.1 0']
    tdl=0
    tdl2=0
    bloom_nt=open(path+"/ntmut_fitness_by_clade.txt",'r')
    lines=bloom_nt.readlines()
    bloom_nt.close()
    bloom_nt_dictionary={}
    bloom_nt_fit={}
    max_expected_count=0
    max_id=0
    for line in lines[1:]:
        nclade,nt_site,nt_mutation,expected_count,actual_count,gene,synonymous,noncoding,four_fold_degenerate,count_terminal,count_non_terminal,mean_log_size,delta_fitness=line.split(',')
        if nclade==eval_clade and abs(float(delta_fitness))>0.0001:
            if not(int(nt_site) in bloom_nt_dictionary):
                if tdl<100:
                    bloom_nt_dictionary[int(nt_site)]=float(expected_count)*np.exp(float(delta_fitness)*tdl)
                else:
                    bloom_nt_dictionary[int(nt_site)]=float(delta_fitness)
            else:
                if tdl<100:
                    bloom_nt_dictionary[int(nt_site)]+=float(expected_count)*np.exp(float(delta_fitness)*tdl)
                else:
                    bloom_nt_dictionary[int(nt_site)]+=float(delta_fitness)
            if tdl<100:
                bloom_nt_fit[nt_mutation[1:]]=float(expected_count)*np.exp(float(delta_fitness)*tdl)
            else:
                bloom_nt_fit[nt_mutation[1:]]=float(delta_fitness)
        #bloom_nt_fit[nt_site+nt]=float(fitness)
        
    
    bloom_aa=open(path+"/aamut_fitness_by_clade.txt",'r')
    lines=bloom_aa.readlines()
    bloom_aa.close()
    bloom_aa_count={}
    bloom_aa_fit={}
    bloom_aa_origin={}
    bloom_aa_full_fit={}

    for line in lines[1:]:
        clade,gene,clade_founder_aa,mutant_aa,aa_site,aa_mutation,expected_count,actual_count,count_terminal,count_non_terminal,mean_log_size,subset_of_ORF1ab,delta_fitness=line.split(',')
        if clade==eval_clade:
            if gene=='S':
               pos=int(aa_site)
               bloom_aa_origin[pos]=clade_founder_aa
               if clade_founder_aa=='*':
                   bloom_aa_origin[pos]='O'
               if mutant_aa=='*':
                   mutant_aa='O'
               fit=float(delta_fitness)
               exp_count=float(expected_count)
               if not(pos in bloom_aa_fit):
                   bloom_aa_fit[pos]={}
                   bloom_aa_count[pos]={}
               bloom_aa_fit[pos][mutant_aa]=fit
               bloom_aa_count[pos][mutant_aa]=exp_count
    for pos in bloom_aa_fit:
        for aa in bloom_aa_fit[pos]:
            if not(pos in bloom_aa_origin):
                bloom_aa_origin[pos]=get_aa(pos,ref_base)
            if aa!=bloom_aa_origin[pos]:
                tb=bloom_aa_fit[pos]
                
                delta=tb[aa]
                if tdl2<100:
                
                    bloom_aa_full_fit['S:'+bloom_aa_origin[pos]+str(pos)+aa]=bloom_aa_count[pos][aa]*np.exp(delta*tdl2)
                else:
                    bloom_aa_full_fit['S:'+bloom_aa_origin[pos]+str(pos)+aa]=delta
    
    return bloom_nt_dictionary,bloom_nt_fit,bloom_aa_origin,bloom_aa_full_fit


def read_ref_seqs(path):
    f=open(path+"/base_variant_seqs.json",'r')
    lines=f.readlines()
    seq_dic={}
    for l in lines:
        ll=json.loads(l)
        seq_dic[ll['variant']]=ll['ref_seq']
    
    return seq_dic

def get_aa(pos,ref_seq):
    start=21563
    ht=ref_seq[start-3+pos*3:start+pos*3]
    result_aa=trans_table[ht]
    return result_aa
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
def evaluate_bloom_single(data_iterator,ref_dic,ref_seq,variants_encoding_dict,k_keys_sorted,k_fits_sorted,k_spike_sorted,bloom_aa_origin):
    nuc_dic={'A':0,"T":1,"C":2,'G':3,'-':4}
    inv_nuc_dic={0:'A',1:'T',2:'C',3:'G',4:'-'}
    data=next(data_iterator)
    if data is None:
        return None
    #print(ref_dic.index)
    data=json.loads(data[0])
    current_seq=copy.deepcopy(ref_seq)
    #print(data)
    var_encode=variants_encoding_dict[data['variant']]
    for i in range(len(var_encode)):
        mut,current_seq=check_mutation(var_encode[i],current_seq,version=1)

    if eval_clade=='23I':
        ref_base=ref_dic['BA.2.86 0']
    if eval_clade=='24A':
        ref_base=ref_dic['JN.1 0']

    sum_single=0
    sum_sp=0
    top_1_nt=0
    top_10_nt=0
    top_100_nt=0
    top_1_mut=0
    top_10_mut=0
    top_100_mut=0
    top_1_sp=0
    top_10_sp=0
    weight=data['weight']
   
    for i in range(len(data['sequence_encoding'])):
        mutation=data['sequence_encoding'][i]
        mutation_pos=(mutation-1)//5+1
        mutation_target=inv_nuc_dic[(mutation-1)%5]
        if current_seq[mutation_pos-1]!='-':
            mutation=str(mutation_pos)+mutation_target
            
            if ref_seq[mutation_pos-1]==ref_base[mutation_pos-1]:
                sum_single+=1
            rank=0
            for j in range(len(k_keys_sorted)):
                pos=k_keys_sorted[j]
                # ensure this is not mutated
                if current_seq[pos-1]==ref_base[pos-1]:
                    rank+=1
                    if pos==mutation_pos:
                        if rank==1:
                            top_1_nt+=1
                        if rank<=10:
                            top_10_nt+=1
                        if rank<=100:
                            top_100_nt+=1
            rank=0
            for j in range(len(k_fits_sorted)):
                mut=k_fits_sorted[j]
                pos=int(mut[:-1])
                if current_seq[pos-1]==ref_base[pos-1]:
                    rank+=1
                    if mutation==mut:
                        if rank==1:
                            top_1_mut+=1
                        if rank<=10:
                            top_10_mut+=1
                        if rank<=100:
                            top_100_mut+=1      
            
            full_mut=get_mutation(current_seq,mutation_pos,mutation_target,anno,trans_table)
            if full_mut!=None:
                for item in full_mut:
                    if 'S:' in item:
                        #ensure this position is not already mutated
                        pos=int(item[3:-1])
                        if not(pos in bloom_aa_origin):
                            bloom_aa_origin[pos]=get_aa(pos,ref_base)
                        if item[2]==bloom_aa_origin[int(item[3:-1])]:
                            sum_sp+=1
                        rank=0
                        for j in range(len(k_spike_sorted)):
                            mut=k_spike_sorted[j]
                            #print(rank,mut)
                            # ensure this is not mutated
                            if item[2]==mut[2] or (item[3:-1]!=mut[3:-1]):
                                rank+=1
                                if item==mut:
                                    if rank==1:
                                        top_1_sp+=1
                                    if rank<=10:
                                        top_10_sp+=1
                              

            current_seq=current_seq[:mutation_pos-1]+mutation_target+current_seq[mutation_pos:]
        else:
            print(mutation_pos,current_seq[mutation_pos-1],ref_seq[mutation_pos-1])
   
    return sum_single,sum_sp,top_1_nt,top_10_nt,top_100_nt,top_1_mut,top_10_mut,top_100_mut,top_1_sp,top_10_sp,weight
        


import heapq
from utils_mutation import *
def evaluate_bloom():
    print("start")
    date="2025-07-16"
    data_dir=data_dir_prefix+"data"+date
    variants_encoding_dict=read_variant_table(data_dir+'/base_variants_v1.json')
    ref_dic=read_ref_seqs(data_dir)
    bloom_nt_dic,bloom_nt_fit,bloom_aa_origin,bloom_aa_fit=read_bloom(data_dir_prefix+bloom_dir,ref_dic)
    k_keys_sorted = heapq.nlargest(200,bloom_nt_dic,key=lambda i:bloom_nt_dic[i])
    #print(bloom_nt_dic)
    k_fits_sorted=heapq.nlargest(200,bloom_nt_fit,key=lambda i:bloom_nt_fit[i])
    k_spikes_sorted=heapq.nlargest(200,bloom_aa_fit,key=lambda i:bloom_aa_fit[i])
    print(k_keys_sorted)
    print(k_fits_sorted)
    print(k_spikes_sorted)
    #for item in k_keys_sorted:
    #    print(item,bloom_nt_dic[item],bloom_nt_fit[str(item)+'T'])
    sum_all=0
    sg=0
    sw=0
    sum_sp_all=0
    top_1_nt_all=0
    top_10_nt_all=0
    top_100_nt_all=0
    top_1_fit_all=0
    top_10_fit_all=0
    top_100_fit_all=0
    top_1_sp_all=0
    top_10_sp_all=0
    top_1_fit_r=0
    top_10_fit_r=0
    top_100_fit_r=0
    top_1_sp_r=0
    top_10_sp_r=0
    valid_date="2025-02-13"
    date_train=['2019-01-01',valid_date]
    date_valid=[valid_date,date]
    if eval_month!=None:
        date_valid=[eval_month+'-01',eval_month+'-31']
   
    
    eval_iterator=data_iterator_sample(data_dir+'/shuffled_seqs_v1',variants_encoding_dict,date_range=date_valid,total_ranks=1,process=False,version=1,evaluation=True)
    i=0
    terminate=True
    ref_seq=read_ref()
    while terminate:
        i+=1
        
        returned_issues=evaluate_bloom_single(eval_iterator,ref_dic,ref_seq,variants_encoding_dict,k_keys_sorted,k_fits_sorted,k_spikes_sorted,bloom_aa_origin)
        if returned_issues is None:
            terminate=False
        else:
            sum_single,sum_sp,top_1_nt,top_10_nt,top_100_nt,top_1_fit,top_10_fit,top_100_fit,top_1_sp,top_10_sp,weight=returned_issues
            if sum_single!=0:
                sum_all+=weight
                sg+=1

                top_1_nt_all+=top_1_nt*weight/sum_single
                top_10_nt_all+=top_10_nt*weight/sum_single
                top_100_nt_all+=top_100_nt*weight/sum_single
                top_1_fit_all+=top_1_fit*weight/sum_single
                top_10_fit_all+=top_10_fit*weight/sum_single
                top_100_fit_all+=top_100_fit*weight/sum_single
                top_1_fit_r+=top_1_fit/sum_single
                top_10_fit_r+=top_10_fit/sum_single
                top_100_fit_r+=top_100_fit/sum_single
            if sum_sp!=0:
                sum_sp_all+=weight
                top_1_sp_all+=top_1_sp*weight/sum_sp
                top_10_sp_all+=top_10_sp*weight/sum_sp
                top_1_sp_r+=top_1_sp/sum_sp
                top_10_sp_r+=top_10_sp/sum_sp
                sw+=1
        if i%5000==0:
            print("processed ",i,' seqs')
    
    #print(sum_all,sum_sp_all,sg,sw,top_1_nt_all,top_10_nt_all,top_100_nt_all,top_1_fit_all,top_10_fit_all,top_100_fit_all)
    #print(top_1_nt_all/sum_all, top_10_nt_all/sum_all, top_100_nt_all/sum_all)

    print("Bloom nuc prediction(weighted) acc: top 1:",top_1_fit_all/sum_all, " top 10:",top_10_fit_all/sum_all, "top 100:", top_100_fit_all/sum_all)
    print("Bloom spike prediction(weighted) acc: top 1:",top_1_sp_all/sum_sp_all," top 10:", top_10_sp_all/sum_sp_all)
    print("Bloom nuc prediction(raw) acc: top 1:",top_1_fit_r/sg," top 10:", top_10_fit_r/sg," top 100:", top_100_fit_r/sg)
    print("Bloom spike prediction(raw) acc: top 1:",top_1_sp_r/sw," top 10:",top_10_sp_r/sw)


evaluate_bloom()
    
    

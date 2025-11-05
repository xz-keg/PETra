import json
from utils import *
import copy
country_list=read_country_list()

import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--date', type=str,default='2025-07-16')
parser.add_argument('--data-dir', type=str,default=None)
args = parser.parse_args()
data_dir=args.data_dir
date=args.date

version=1
w=open("lr_data"+date+".txt",'r')
q=w.readlines()
lineage_mutations={}
for line in q:
    if len(line)>5:
        lineage,mutation=line.split('[',1)
        #print(lineage)
        if lineage.strip()!='B 0':
            
            mut_profile=json.loads('['+mutation.strip().replace("'",'"'))
            print(line,lineage)
            lineage_text,lineage_id=lineage.strip().split()
            lineage_mutations[lineage.strip()]=copy.deepcopy(mut_profile)
w.close()
#print(lineage_mutations)

def dump_mutation_ids():
    ww=read_mutation_ids(dt=date)
    print(ww,len(ww))
    f=open("mutation_list"+date+".txt",'w')
    for i in range(300000):
        if i in ww:
            print(json.dumps(ww[i]),file=f)
        else:
            
            print(i)
    f.close()
    return ww
dps=dump_mutation_ids()
def read_mutation_ids_current():
    f=open("mutation_list"+date+".txt",'r')
    li=f.readlines()
    f.close()
    return li
mutation_ids=read_mutation_ids_current()
print(len(mutation_ids))
#print(mutation_ids)
lineage_actual={'B 0':[]}
k=0
def gather_mutations(lineage_desc):
    if lineage_desc in lineage_actual:
        return copy.deepcopy(lineage_actual[lineage_desc])

    mut_profile=lineage_mutations[lineage_desc]
    parent_desc=mut_profile[0]+' '+str(mut_profile[1])
    add_mutations=copy.deepcopy(mut_profile[2])
    expressed_mutations=[]
    for mut in add_mutations:
        #print(mut,len(mutation_ids))
        mut_profile=json.loads(mutation_ids[mut])
        if mut_profile['mutation_id']!=mut:
            print(mut_profile,mut)
        if mut_profile['gene']=='nt':
            # only consider nuc mutations
            mutation=mut_profile['previous_residue']+str(mut_profile['residue_pos'])+mut_profile['new_residue']
            expressed_mutations.append(mutation)
    mm=gather_mutations(parent_desc)
    
    mm.extend(expressed_mutations)
    final_mutations=[]
    already=[]
    
    for i in range(len(mm)):
        
        mut=mm[i]
        pos=int(mut[1:-1])
        if not(pos in already):
            already.append(pos)
            for j in range(i,len(mm)):
                if int(mm[j][1:-1])==pos:
                    mut=mut[:-1]+mm[j][-1]
            if not(mut[0]==mut[-1]):
                if 'X' in mut:
                    print(mut,lineage_desc)
                final_mutations.append(mut)
        

    lineage_actual[lineage_desc]=copy.deepcopy(final_mutations)
    print(lineage_desc, len(final_mutations))
    return final_mutations

print("now")
lineage_mutations['B 0']=['B',0,[]]
for lineage in lineage_mutations:
    lmut=gather_mutations(lineage)
    #print(len(lineage_actual))
    
trans_table=read_table()
ref_seq=read_ref()
variant_mutation_dic={}
variant_mutation_dic['B']=[]
anno=read_designation(variant_mutation_dic)

for lineage in variant_mutation_dic:
    # adjust: 28273- -> 28271-
    if 'A28273-' in variant_mutation_dic[lineage] and not('A28271-' in variant_mutation_dic[lineage]):
        variant_mutation_dic[lineage].remove('A28273-')
        variant_mutation_dic[lineage].append('A28271-')

refers={}
#diff_file=open("difference"+date+".txt",'w')

def read_problematic_sites():
    try:
        site_f=open("problematic_pos"+date+".txt",'r')
        lines=site_f.readlines()
    except:
        print("problematic pos file not found.")
        return {}
    problematic_sites={}
    for line in lines: 
        variant,pos=line.split()
        if not(variant in problematic_sites):
            problematic_sites[variant]=[pos]
        else:
            problematic_sites[variant].append(pos)
    return problematic_sites

problematic_sites=read_problematic_sites()

print(problematic_sites)

excess_mutations_table={}
unexpected_mutations_table={}
for lineage in lineage_mutations:
    referring_lineage=''
    lin_now=lineage
    while not(referring_lineage in variant_mutation_dic):
        ll=lin_now.split(' ')[0]
        referring_lineage=ll
        if ('_' in ll and not('dropout' in ll)):
            referring_lineage=ll.split('_')[0]
        print(lin_now,lineage_mutations[lin_now])
        lin_now=lineage_mutations[lin_now][0]+' '+str(lineage_mutations[lin_now][1])
        #lin_now=ll+' '+str(lineage_mutations[lin_now][1])
        #print(lin_now)
    refers[lineage]=referring_lineage
    # first, fix out refers of each lineage.
#print(refers)

output_set=['proposed467 0','BA.2_10507_18744 0','BA.2.86_dropout 0','BA.4_dropout 0','BA.5_dropout 0']
for lineage in lineage_mutations:

    lmut=lineage_actual[lineage]
    referring_mutations=variant_mutation_dic[refers[lineage]]

    excess_mutations=[]
    unexpected_mutations=[]
    problematic_pos=[]
    # problematic site shall be something that is traceable
    
    current_ref=lineage.split(' ')[0]
    
    if current_ref in problematic_sites:
        print('current ref hit',current_ref,problematic_sites[current_ref])
        problematic_pos=problematic_sites[current_ref]
        

    for mutation in lmut:
        if not(mutation in referring_mutations):
            if not(mutation[:-1]+'-' in referring_mutations):
                if not(int(mutation[1:-1]) in problematic_pos):
                    if 'X' in mutation:
                        print(mutation,lineage)
                    excess_mutations.append(mutation)
    for mutation in referring_mutations:
        if not(mutation in lmut):
            # for position 1 and 44, just ignore
            if int(mutation[1:-1])>50:
                unexpected_mutations.append(mutation)
    
    excess_mutations_table[lineage]=copy.deepcopy(excess_mutations)
    unexpected_mutations_table[lineage]=unexpected_mutations
    if lineage in output_set: 
        print(lineage, refers[lineage],excess_mutations,unexpected_mutations,len(lmut))
    #
  
    # edit mutations for excess/unexpected mutations


f=open('mutation_data'+date+'.txt','r')
ff=f.readlines()
if data_dir is None:
    data_dir="../data"+date

import os
hasit=os.listdir("../")
if not('data'+date in hasit):
    os.mkdir(data_dir)
existing_data=os.listdir(data_dir)


def process_lineages(encoding_version=0):
    ref_seqs={}
    outfile=open(data_dir+"/base_variants_v"+str(encoding_version)+".json",'w')
    outfile2=open(data_dir+"/base_variant_seqs.json",'w')
    max_unexpected_mutations_table=0
    for base_variant in refers:
        # two ways: lineage_actual(from tree)+ unexpected_mutations (from designation)
        # or variant_mutations(from designation)+excess_mutations(from tree)

        # use way 2: variant_mutations+excess_mutations
        referring_lineage=refers[base_variant]
        base_mutations=copy.deepcopy(variant_mutation_dic[referring_lineage])
        
        if base_variant in output_set:
            print(base_variant, len(base_mutations),base_mutations,len(excess_mutations_table[base_variant]),excess_mutations_table[base_variant])
        base_mutations.extend(copy.deepcopy(excess_mutations_table[base_variant]))
        reference_here=ref_seq
        variant_encoding=[]
        sequence_encoding=[]
        existing_positions=[]
        i=0
        while i<len(base_mutations):
            item=base_mutations[i]
            isset=True
            if int(item[1:-1])<50:
                base_mutations.remove(item)
                isset=False
            if int(item[1:-1]) in existing_positions:
                for itemx in base_mutations[:i]:
                    if int(itemx[1:-1])==int(item[1:-1]):
                        if itemx[0]==item[-1]:
                            itemx=item[0]+itemx[1:]
                        else:
                            if itemx[-1]!='-':
                                itemx=itemx[:-1]+item[-1]
                            
                        base_mutations.remove(item)
                        isset=False
            if isset:           
                existing_positions.append(int(item[1:-1]))
                i+=1

        for mutation in base_mutations:
            encoding,new_ref=mutation_encoding(reference_here,mutation,anno,trans_table,version=encoding_version)
            if encoding is not None:
                reference_here=new_ref
                variant_encoding.extend(encoding)
        ref_seqs[base_variant]=reference_here
        dic={}
        dic['variant']=base_variant
        dic['encoding']=variant_encoding
        print(json.dumps(dic),file=outfile)
        dic2={}
        dic2['variant']=base_variant
        dic2['ref_seq']=reference_here
        print(json.dumps(dic2),file=outfile2)

    outfile.close() 
    outfile2.close()
    return ref_seqs
def read_reference_seqs():
    ref_seqs={}
    f=open(data_dir+"/base_variant_seqs.json",'r' )
    lf=f.readlines()
    for line in lf:
        extract=json.loads(line)
        variant=extract['variant']
        ref_seq_here=extract['ref_seq']
        ref_seqs[variant]=ref_seq_here
    f.close()
    return ref_seqs
force_process=True
if force_process or not('base_variants.json' in existing_data):
    ref_seqs=process_lineages(encoding_version=version)
else:
    ref_seqs=read_reference_seqs()



data_filename=0
while str(data_filename)+".json" in existing_data:
    data_filename+=1
# 2000=1 batch
current_count=0
#print(country_list)
if not('seqs_v'+str(version) in existing_data):
    os.mkdir(data_dir+'/seqs_v'+str(version))
existing_data=os.listdir(data_dir+"/seqs_v"+str(version))
output_file=open(data_dir+"/seqs_v"+str(version)+"/"+str(data_filename)+".json",'w')
start_id=0
#print(ff[0])
#print(ff[0][0:99])

#print(json.loads(ff[0].replace("'",'"').replace(": ",":")))
printed=[]
for i in range(start_id,len(ff)):
    #if i%100==0:
    #    print(i,ff[i])
    try:
        properties=json.loads(ff[i].replace("d'",'d').replace("'",'"'))
    except:
        print(ff[i],ff[i][:210])
        
    mutation_desc=properties['mutation_desc']
    base_variant=mutation_desc[0]+' '+str(mutation_desc[1])
    additional_mutations=mutation_desc[2]
    prt=False
    if len(additional_mutations)==0 and base_variant!='B 0' and not(('X' in base_variant) and not('.' in base_variant)):
        #back to previous stage
        brc=lineage_mutations[base_variant]
        
        new_base=brc[0]+' '+str(brc[1])
        additional_mutations=copy.deepcopy(brc[2])
       
        for item in unexpected_mutations_table[base_variant]:
            if not(item in unexpected_mutations_table[new_base]):
                additional_mutations.append(item)
                prt=True
        base_variant=new_base
        if prt and not(base_variant in printed):
            print(mutation_desc,brc,additional_mutations)
            printed.append(base_variant)

    referring_lineage=refers[base_variant]
    base_mutations=variant_mutation_dic[referring_lineage]
    base_mutations.extend(unexpected_mutations_table[base_variant])
    expressed_mutations=[]
    for mut in additional_mutations:
        if not(isinstance(mut,int)):
            expressed_mutations.append(mut)
        else:
            mut_profile=json.loads(mutation_ids[mut])
            if mut_profile['mutation_id']!=mut:
                print(mut_profile,mut)
            if mut_profile['gene']=='nt':
                # only consider nuc mutations
                mutation=mut_profile['previous_residue']+str(mut_profile['residue_pos'])+mut_profile['new_residue']
                expressed_mutations.append(mutation)
    
    reference_here=ref_seqs[base_variant]
    time=properties['meta_date']
    location=properties['location']
    location_time_encoded=location_time_encoding(location,time,country_list,version=version)
    variant_encoding=[]
    sequence_encoding=[]
  
    for mutation in expressed_mutations:
        encoding,new_ref=mutation_encoding(reference_here,mutation,anno,trans_table,version=version)
        reference_here=new_ref
        if not(encoding is None):
            sequence_encoding.extend(encoding)
    profile={}
    profile['location_time_encoding']=location_time_encoded
    profile['variant']=base_variant
    profile['sequence_encoding']=sequence_encoding
    profile['weight']=properties['weight']
    print(json.dumps(profile),file=output_file)
    current_count+=1
    if current_count>=2000:
        print("Finshed encoding ",i,' seqs')
        current_count=0
        existing_data=os.listdir(data_dir+"/seqs_v"+str(version))
        while str(data_filename)+".json" in existing_data:
            data_filename+=1
        output_file.close()
        output_file=open(data_dir+"/seqs_v"+str(version)+"/"+str(data_filename)+".json",'w')

output_file.close()


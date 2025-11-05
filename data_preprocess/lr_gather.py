import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--date', type=str,default='2025-07-16')

args = parser.parse_args()
date=args.date

import json
import copy

w=open("lr_data"+date+".txt",'r')
q=w.readlines()
lineage_mutations={}
for line in q:
    lineage,mutation=line.split('[',1)
    #print(lineage)
    if lineage.strip()!='B 0':
        
        mut_profile=json.loads('['+mutation.strip().replace("'",'"'))
        #print(line,lineage)
        lineage_text,lineage_id=lineage.strip().split()
        lineage_mutations[lineage.strip()]=copy.deepcopy(mut_profile)
w.close()

lineage_count={}
lineage_general_parent={}
lineage_general_mutations={}
for item in lineage_mutations:
    variant=item.split(' ')[0]
    if not(variant in lineage_count):
        lineage_count[variant]=1
        lineage_general_parent[variant]=[lineage_mutations[item][0],lineage_mutations[item][1]]
        lineage_general_mutations[variant]=lineage_mutations[item][2]
    else:
        lineage_count[variant]+=1
        this_parent=[lineage_mutations[item][0],lineage_mutations[item][1]]
        if this_parent!=lineage_general_parent[variant]:
            lineage_general_parent[variant]=None
        this_mutation=lineage_mutations[item][2]
        i=0
        while i<len(lineage_general_mutations[variant]):
            mut=lineage_general_mutations[variant][i]
            if not(mut in this_mutation):
                lineage_general_mutations[variant].remove(mut)
                i-=1
            i+=1

outfile=open("lr_data"+date+".txt",'a')
print('',file=outfile)
for item in lineage_count:
    if lineage_count[item]>=2:

        if lineage_general_parent[item] is not None:
            if len(lineage_general_mutations[item])>0:
                lineage_output=copy.deepcopy(lineage_general_parent[item])
                lineage_output.append(copy.deepcopy(lineage_general_mutations[item]))

                print(item,'c',lineage_output,file=outfile)
outfile.close()
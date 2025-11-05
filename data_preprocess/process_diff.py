import json
from utils import *
import copy
country_list=read_country_list()
import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--date', type=str,default='2025-07-16')

args = parser.parse_args()
date=args.date

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
    ww=read_mutation_ids(date)
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
diff_file=open("difference"+date+".txt",'w')

def read_problematic_sites():
    return {}
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
''' 
problematic_sites={'JH.2':[23013],'XBM':[12160],'BA.5.2.2':[29510],'XBG':[12160],'XBT':[12160],
'XAV':[12160,27889],'XAN':[12160],'XAZ':[27889],'XDD':[26610,6183,9142],'XDS':[6183,9142],'XDR':[6183,9142],
'XDN':[26610],'XDQ':[24378,26610],'XDT':[24378,26610,6183],'XCK':[26858,27915],'XCD':[26858,27915],'XBW':[26858,27915],
'XBB.1.5.85':[22281],'EG.6.1.1':[23013],'EG.11':[19326],'FL.4.4':[22995],'FL.13.4.1':[23013],'FL.30':[23013],
'XBV':[17859,19326],'XBF':[3927,3796,4586,5183],'B.1.214.3':[28273,28274],'B.1.189':[4582],'B.1.237':[28854],
'B.1.177.44':[29366],'B.1.576':[335],'B.1.170':[28977],'B.1.640.1':[28005,28093,27925],'B.1.456':[1102,4124],
'B.1.36.10':[6681],'B.1.596':[10319,28869,28472,25244],'B.1.1.8':[6573],'B.1.560':[28881,28882,28883],
'P.5':[1627,3617,5180,9929,10888,12664,21614,23542,24374,27807,28311],'XC':[28461],
'P.4':[20745,25708],'BA.1.23':[10741],'XAW':[26530,22898],'XD':[26530],'XAY.1.1.2':[27575],
'XAK':[23040],'XBS':[26275,22898],'XBR':[22190,22331,22577,26275],
'XBP':[26275,22577,22898,22331],'XCA':[26275,22898],'BL.5':[29540],'CJ.1.3':[261],
'XBH':[4586,3796,3927,12444,5183,15451],'XBU':[22898],'EP.2':[3037],'EP.1.1':[3037],
'GQ.1.1':[10029],'CH.1.1.21':[27576],'BA.2.80':[22580],'JY.1':[22995],'XBB.1.33':[22688],
'XCM':[22109,23031,22896,22895,22664,22317],'FW.1.1.1':[27383,27384],'JC.2':[22578],
'XC':[28461]
}
'''

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
    current_lineage=lineage
    current_ref=refers[lineage]
    
    if current_ref in problematic_sites:
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
    
    excess_mutations_table[lineage]=excess_mutations
    unexpected_mutations_table[lineage]=unexpected_mutations
    if lineage in output_set: 
        print(lineage, refers[lineage],excess_mutations,unexpected_mutations,len(lmut))
    #
    if (lineage.split(' ')[-1]=='0' and not(lineage.replace(' 0',' 1') in lineage_mutations) or lineage.split(' ')[-1]=='c') and len(excess_mutations)>0 and lineage.split(' ')[0]==refers[lineage]:
       print(lineage,refers[lineage], excess_mutations,unexpected_mutations,file=diff_file)
    # edit mutations for excess/unexpected mutations
diff_file.close()
f=open('mutation_data'+date+'.txt','r')
ff=f.readlines()


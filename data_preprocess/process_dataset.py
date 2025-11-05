# to make a dataset of nodes. 
# property of nodes: country, collection date, weight, mutation path

import json
import numpy as np
import copy

from utils import *

node_by_id={}
weightdic={}
mutation_dic={}
weightsum={}
lcount=0


import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--date', type=str,default='2025-07-16')

args = parser.parse_args()
date=args.date
yy,mm,dd=date.split('-')
outprfile=open("mutation_data"+date+".txt",'w')
outlrfile=open("lr_data"+date+".txt",'w')


trans_table=read_table()
ref_seq=read_ref()
variant_mutation_dic={}
anno=read_designation(variant_mutation_dic)

ref_dic={}



vanish_keys=['cat','env','mesocricetusauratus','mouse','bird','rodent','mink','dog','lion','deer',
             'neovisonvison','canisfamiliaris','gorilla','tiger','ferret','snowleopard',
             'mink','musmusculus','puma','canine','asiaticlion','otter','caprahircus','ovisaries',
             'susscrofa','canislupus','hippo','hamster','odocoileusvirginianus','hyena','lacertilia',
             'bostaurus','caoti','monkey','armadillo','leopard','syrianhamster','dogdomestic','country',
             'catdomestic','minkamerican','whitetaileddeer']

mutation_for_lineage={}



def trans(mutation):
    baseline=mutation[0]
    for item in mutation[2]:
        mut=mutation_dic[item]
        desc=mut["previous_residue"]+str(mut["residue_pos"])+mut["new_residue"]
        if mut['type']=='aa':
            desc=mut['gene']+':'+desc
        if mut['gene']!='ORF1ab':
            baseline=baseline+'+'+desc
    return baseline

def check_child(la,lb):
    # check if lb is a child of la. 
    sta=la.split('.')[0].strip()
    stb=lb.split('.')[0].strip()
    if sta==stb:
        if la.strip() in lb.strip():
            return True
        else:
            return False
    
    if 'X' in stb:
        return False

    if len(sta)<len(stb)  or (len(sta)==len(stb) and sta<stb) or ('X' in sta and len(la.split('.'))==3):
        return True
    
    return False
    

def track_tree(node_id,mutation_desc):
    global lcount
    lcount+=1
    if lcount%1000000==0:
        print("lcount=",lcount)
    if mutation_desc[2] is None:
        print(node_id,mutation_desc,'Noneprinter')
    node=node_by_id[node_id]
    lineage=node["meta_pango_lineage_usher"]
    
    def search_util_have(num):
        lineage_set=[]
        node=node_by_id[num]
        if len(node['meta_pango_lineage_usher'])!=0:
            return [node['meta_pango_lineage_usher']]
        if not('child_id') in node:
            print(node)
            return lineage_set
        for item in node['child_id']:

            lst=search_util_have(item)
            if len(node_by_id[item]['mutations'])==0:
                # if there is a direct point, get direct answer
                return lst
            for item in lst:
                if not(item in lineage_set):
                    lineage_set.append(item)
        return lineage_set
    # do not aggregate, as usher may have weird placement issues. 
    
    lineage=''
    if len(lineage)==0:
        # try to find which lineage its children belong to 
        lineage_set=search_util_have(node_id)
        if len(lineage_set)==1:
            lineage=lineage_set[0]
        
    # back to traditional lineage+mutation mode. 

    if (lineage!=mutation_desc[0]) and (len(lineage)>0):
        if node["mutations"] is not None:
            mutation_desc[2].extend(node["mutations"])
        if lineage in mutation_for_lineage:
            mutation_for_lineage[lineage].append(mutation_desc)
        else:
            mutation_for_lineage[lineage]=[mutation_desc]
        
        
        mutation_new=[lineage,len(mutation_for_lineage[lineage])-1,[]]
        print(lineage,len(mutation_for_lineage[lineage])-1,mutation_desc,node["mutations"])
        print(lineage,len(mutation_for_lineage[lineage])-1,mutation_desc,file=outlrfile)
    else:
        
        if not("mutations" in node) or node["mutations"] is None:
            node["mutations"]=[]
    
        mutation_new=copy.deepcopy(mutation_desc)
        mutation_new[2].extend(node["mutations"])
        if mutation_new[2] is None:
            print(node_id,mutation_desc,node['mutations'],'newnoneprinter')
    weight=0
    count=0
    node_by_id[node_id]["mutation_desc"]=copy.deepcopy(mutation_new)
    wd={}
    ct={}
    ald={}
    
    if node["is_tip"]==True:

        if ("location" in node) and ("yymm" in node) and int(node["yymm"])>201900:
            weight=weightdic[node["location"]][node["yymm"]]
            wd[node["yymm"]]=weight
            ct[node["yymm"]]=1
            nw=copy.deepcopy(node)
            nw['weight']=weight
            nw['mutation_desc']=mutation_desc
            print(json.dumps(nw),file=outprfile)
        return wd,ct,ald

    else:
        if 'child_id' in node:
            for item in node["child_id"]:
                #print(item,mutation_new,'exploreprinter')
                nw,nc,ad=track_tree(item,copy.deepcopy(mutation_new))
                for itm in nw:
                    if not(itm in wd):
                        wd[itm]=0
                        ct[itm]=0
                        ald[itm]=0
        
                    wd[itm]+=nw[itm]
                    ct[itm]+=nc[itm]
                    if itm in ad:
                        ald[itm]+=ad[itm]
                


    pr=False
    
    for item in wd:
        wdald=wd[item]-ald[item]
        
        fac=wd[item]/(ald[item]+0.01)
        if fac>2:
            if (wdald>0.01*weightsum[item]) or (wdald>0.005*weightsum[item] and int(item)>202100) or (wdald>0.002*weightsum[item] and int(item)>202200):
                pr=True
    if pr:
        mut=trans(mutation_new)
        #print(mut)
        #print(mut,file=outfile)
        for yy in range(2025,2026):
            for mm in range(1,10):
                item=str(yy*100+mm)
                if item in wd:
                    #print(item,wd[item]/weightsum[item],ct[item],file=outfile)
                    ald[item]=wd[item]
                    if wd[item]/weightsum[item]>0.002:
                        print(item,round(wd[item]/weightsum[item]*1000,2),ct[item])
    
    return wd,ct,ald
def process_usher():
    seqcountdic={}
    global weightdic
    weightdic=json.load(open("weight_dic_proc_"+mm+dd+".json",'r'))
    
    nums=0
    presented_ctr=[]
    f=open('usher_tree'+date+'.txt','r')
    lines=f.readlines()
    f.close()
    l0=json.loads(lines[0])
    mutations=l0['mutations']
    for item in mutations:
        ids=item["mutation_id"]
        mutation_dic[ids]=item
   

    countries={}
    existing_bad=[]
    num=0
    root_set=[]
    count=0
    for item in lines[2:]:
        #print(item)
        count+=1
        if count%1000000==0:
            print(count)
        vi=json.loads(item)
        if vi["is_tip"]==False:
            vi["seq_count"]=0
            vi["weighted_count"]=0
            vi["child_id"]=[]
        node_by_id[vi["node_id"]]=vi
        if vi["parent_id"]==vi["node_id"]:
            root_set.append(vi["node_id"])
    print("finished loading nodes")
    count=0
    for nde in node_by_id:
        count+=1
        if count%1000000==0:
            print(count)
        node=node_by_id[nde]
        if node["is_tip"]==True:
            if len(node["mutations"])==0:
                nl=node_by_id[node["parent_id"]]
                if len(nl["meta_pango_lineage_usher"])==0:
                    node_by_id[node["parent_id"]]["meta_pango_lineage_usher"]=node["meta_pango_lineage_usher"]

            if ("location" in node) and ("yymm" in node):
                weight=weightdic[node["location"]][node["yymm"]]
                if not(node["yymm"] in weightsum):
                    weightsum[node["yymm"]]=0
                weightsum[node["yymm"]]+=weight

        if not(node["node_id"] in root_set):
            node_by_id[node["parent_id"]]["child_id"].append(node["node_id"])
    print("finished initializing nodes")
    print(weightsum)
    print(root_set)
    for item in root_set:
        
        weight,count,pr=track_tree(item,['',0,[]])

        
    return 0

process_usher()

outprfile.close()
outlrfile.close()

from urllib import request

import json
import time
# for each lineage, query cov-spectrum to decide whether a mutation exists or not

import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--date', type=str,default='2025-07-16')

args = parser.parse_args()

date=args.date


def decide_mutation(mut,lineage):
    
    url_prefix="https://lapis.cov-spectrum.org/gisaid/v2/sample/aggregated?dateFrom=2020-02-19&dateTo="+date
    query_mut=mut[1:]
    query_lineage=lineage
    url_postfix="&host=Human&accessKey=9Cb3CqmrFnVjO3XCxQLO6gUnKPd"
    url=url_prefix+"&nucleotideMutations="+query_mut+"&nextcladePangoLineage="+query_lineage+'*'+url_postfix
    try:
        req=request.urlopen(url,timeout=3)
    except:
        # retry first
        time.sleep(1)
        try:
            req=request.urlopen(url,timeout=3)
        except:
            print(url)
            return 2

    res_text=req.readlines()
    #print(json.loads(res_text[0])['data'])
    number_of_seqs_new=json.loads(res_text[0])['data'][0]['count']
    query_mut=mut[1:-1]+mut[0]
    url=url_prefix+"&nucleotideMutations="+query_mut+"&nextcladePangoLineage="+query_lineage+'*'+url_postfix
    try:
        req=request.urlopen(url,timeout=3)
    except:
        time.sleep(1)
        try:
            req=request.urlopen(url,timeout=3)
        except:
            print(url)
            return 2

    res_text=req.readlines()
    number_of_seqs_past=json.loads(res_text[0])['data'][0]['count']
    url=url_prefix+"&nextcladePangoLineage="+query_lineage+'*'+url_postfix
    try:
        req=request.urlopen(url,timeout=3)
    except:
        time.sleep(1)
        try:
            req=request.urlopen(url,timeout=3)
        except:
            print(url)
            return 2
    res_text=req.readlines()
    total_seqs=json.loads(res_text[0])['data'][0]['count']
    print(mut,lineage, number_of_seqs_past,number_of_seqs_new,total_seqs)
    if (number_of_seqs_past*10<number_of_seqs_new and number_of_seqs_new*2>total_seqs):
        return 1
    if (number_of_seqs_new*10<number_of_seqs_past and number_of_seqs_past*2>total_seqs):
        return 0
    
    return 2 

diff_file=open("difference"+date+".txt",'r')
lines=diff_file.readlines()
diff_file.close()
out_file=open("problematic_pos"+date+".txt",'w')
out_file2=open("problematic_unexpected"+date+".txt",'w')
out_file3=open("example_seqs",'w')
for l in lines:
    lineage=l.split()[0]
    mutsp=l.split('[')
    excess_muts=mutsp[1].replace(']','').replace(' ','').replace("'","").split(',')
    unexpected_muts=mutsp[2].replace(']','').replace(' ','').replace("'","").split(',')
    for excess_mut in excess_muts:
        decision=decide_mutation(excess_mut,lineage)
        if decision==0:
            print(lineage,excess_mut[1:-1],file=out_file)
        #if decision==2:
        #    print(lineage, excess_mut,"need manual configuration")
    '''
    for unexpected_mut in unexpected_muts:
        decision=decide_mutation(excess_mut,lineage)
        if decision==1:
            print(lineage,excess_mut[1:-1],file=out_file2)
        if decision==2:
            print(lineage, excess_mut,"need manual configuration")
    '''
out_file.close()
out_file2.close()
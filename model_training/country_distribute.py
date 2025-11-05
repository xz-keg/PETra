from utils import *
import os
data_dir_prefix=""
country_list=read_country_list()
inv_country_list={}
for item in country_list:
    inv_country_list[country_list[item]]=item

developed_countries=['austria','belgium','bulgaria','croatia','cyprus','czechrepublic','denmark',
                     'estonia','finland','france','germany','greece','hungary','iceland','ireland',
                     'italy','latvia','lithuania','luxembourg','malta','netherlands','norway','poland',
                     'portugal','romania','slovakia','slovenia','spain','sweden','switzerland','unitedkingdom',
                     'canada','usa','australia','japan','southkorea','newzealand']
least_developed_countries=['angola','benin','burkinafaso','burundi','centralafricanrepublic','chad','comoros',
                           'democraticrepublicofthecongo','djibouti','eritrea','ethiopia','gambia','guinea','guineabissau','lesotho',
                           'liberia','madagascar','malawi','mali','mauritania','niger','mozambique','rwanda',
                           'senegal','sierraleone','somalia','southsudan','sudan','tanzania','togo','uganda',
                           'zambia','haiti','afghanistan','bangladesh','cambodia','timorleste','laos',
                           'myanmar','nepal','yemen','kiribati','solomonislands','tuvalu']

for item in developed_countries:
    if not(item in country_list):
        print(item)
for item in least_developed_countries:
    if not(item in country_list):
        print(item)

        

def data_count(data_dir):
   
    count_dict={}
    total_dict={}
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
        if ('.json'==fname[-5:]) :
                
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
                seq=json.loads(data[0])
                country_id=seq['location_time_encoding'][0]-149600
                month=seq['location_time_encoding'][3]-149990
                country=inv_country_list[country_id]
                category='developing_countries'
                if country in developed_countries:
                    category='developed_countries'
                if country in least_developed_countries:
                    category='least_developed_countries'
                if not (month in count_dict):
                    count_dict[month]={}
                if not (category in count_dict[month]):
                    count_dict[month][category]={'count':0,'weight':0}
                count_dict[month][category]['count']+=1
                count_dict[month][category]['weight']+=seq['weight']
                if not(month in total_dict):
                    total_dict[month]={'count':0,'weight':0}
                total_dict[month]['count']+=1
                total_dict[month]['weight']+=seq['weight']


                current_pos+=1
                total_samples+=1
                if total_samples%1000000==0:
                    print("processed ",total_samples,' samples.')


    return total_dict,count_dict
date1='2025-07-16'

drange2=['2019-01-01',date1]
data_dir=data_dir_prefix+date1+'/seqs_v1'
total_dict,clade_count_dict=data_count(data_dir)
outfile=open("country_count_0716.txt",'w')
for month in clade_count_dict:
    yy=(month-1)//12+2019
    mm=(month-1)%12+1
    print(yy,mm,'overall',total_dict[month]['count'],total_dict[month]['weight'],file=outfile)
    for clade in clade_count_dict[month]:
        print(yy,mm,clade,clade_count_dict[month][clade]['count'],clade_count_dict[month][clade]['weight'],file=outfile)

outfile.close()




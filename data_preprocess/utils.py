import json
import copy
import random
date="2025-07-16"
variant_mutation_dic={}

def designation_browser(current_node,current_mut,variant_mutation_dic):
    mut=[]
    if 'branch_attrs' in current_node:
        #print(current_node['branch_attrs'])
        if 'nuc' in current_node['branch_attrs']['mutations']:
            mut=current_node['branch_attrs']['mutations']['nuc']
    all_mutations=copy.deepcopy(current_mut)
    for item in mut:
        nuc_pos=int(item[1:-1])
        already=False
        for item2 in current_mut:
            if int(item2[1:-1])==nuc_pos:
                already=True
                correct_mut=item2[0]+item[1:]
                all_mutations.remove(item2)
        if not(already):
            all_mutations.append(item)
        else:
            if correct_mut[0]!=correct_mut[-1]:
                all_mutations.append(correct_mut)
    name=current_node['name']
    if not("NODE" in name):
        variant_mutation_dic[name]=all_mutations
    if 'children' in current_node:
        for child in current_node['children']:
            w=designation_browser(child,all_mutations,variant_mutation_dic)

    return 0

def read_designation(variant_mutation_dic):
    w=open("des.json",'r')
    q=json.load(w)
    w.close()
    sp=designation_browser(q['tree'],[],variant_mutation_dic)
    
    anno=q['meta']['genome_annotations']
    anno['ORF9b']={'start':28284,'end':28577}
    anno['ORF9c']={'start':28734,'end':28955}
    anno['ORF3c']={'start':25457,'end':25582}
    anno['ORF3b']={'start':25814,'end':25882}
    anno['ORF3d']={'start':25524,'end':25697}
    anno['ORF3d-2']={'start':25968,'end':26069}
    anno['ORF0']={'start':107,'end':136}
    anno['ORF10']={'start':29558,'end':29674}
    return anno

def read_ref():
    ref=open("reference_seq.txt",'r')
    q=ref.readlines()
    seq=''
    for l in q:
        for w in l:
            if w in ['a','t','c','g']:
                seq=seq+w
    return seq.upper()

def read_table():
    fs=open("table.txt",'r')
    flines=fs.readlines()
    trans_table={}
    for l in flines:
        linsp=l.split()
        encoding=linsp[0]
        translated=linsp[2]
        trans_table[encoding]=translated
    return trans_table 

def compute_ref_dic(ref_seq,mut_dic):
    designated_refs={} #4000*30000
    for lineage in mut_dic:
        designated_refs[lineage]=ref_seq
        for mut in mut_dic[lineage]:
            pos=int(mut[1:-1])
            endmut=mut[-1]
            designated_refs[lineage]=designated_refs[lineage][:pos-1]+endmut+designated_refs[lineage][pos:]
    return designated_refs

# input: location emb, date emb, designated mutation emb, new mutation emb, weight
# output: mutation

def read_mutation_ids(dt=date):
    f=open('usher_tree'+dt+'.txt','r')
    lines=f.readlines()
    f.close()
    l0=json.loads(lines[0])
    mutations=l0['mutations']
    mutation_dic={}
    for item in mutations:
        ids=item["mutation_id"]
        mutation_dic[ids]=item
    f.close()
    return mutation_dic




def mutation_encoding(current_ref,mutation,anno,trans_table,version=0):
    # encoding criteria: 
    # mutation is in form of X0000Y 
    # 0: nothing 

    #  version=0: pos: 1-29903 
    # 29904-29923: A->T  A->C  A->G  T->A T->C  T->G  C->A C->T C->G G->A G->T G->C A->- T->- C->- G->-
    # 29924-29935 : annos   (Orf1a, Orf1b, S, Orf3a, E,M, Orf6, Orf7a, Orf7b, Orf8, N, Orf10 )
    # 29936-29942 : overlap_annos (Orf0, Orf3b, Orf3c, Orf3d, Orf3d-2, Orf9b, Orf9c)
    # 29943-33943 : position of related 1-4401
    # 33944-34405 : 21*22=462 transition map
    # 34406-183920 : 29903*5=149515 mutation profiles
    # [mut, pos, nuc_mut, anno, anno_pos, anno_mut,overlap_anno, overlap_anno_pos, overlap_anno_mut ]
   
    # version=1: mutation: 1-149515
    # [mut]
    # version=2: pos:1-29903 aa_pos/overlap_pos: 29904-41000 mutation: 41000-190514
    #[0 0 0 0 mut pos aa_pos overlap_pos]
    # version=3: pos:1-29903 , 29904-29923: nut_mut 
    # 29924-42000: aa_pos/overlap_aa_pos
    # 42000-42500: aa_mut
    # 42500-192014: mutation
    # encoding:[0 0 0 0 0 0 0 mut pos nuc_mut aa_pos aa_mut overlap_aa_pos overlap_aa_mut]
    # label:[mut pos nuc_mut aa_pos aa_mut overlap_aa_pos overlap_aa_mut, 0 0  0 0 0 0 0 ]



    pos=int(mutation[1:-1])
    # we don't consider restoration of -s. 
    current_nuc=current_ref[pos-1]
    if current_nuc=='-':
        return None,current_ref
    nuc_dic={'A':0,"T":1,"C":2,'G':3,'-':4}
    current_nuc=nuc_dic[current_nuc]
    target_nuc=nuc_dic[mutation[-1]]
    if current_nuc==target_nuc:
        return None,current_ref
    new_ref=current_ref[:pos-1]+mutation[-1]+current_ref[pos:]
    
    encoding_tensor=[]
    if version==1:
        return [(pos-1)*5+target_nuc+1],new_ref
    
    version_length={0:9, 1:1, 2:4, 3:7}
    for i in range(version_length[version]):
        encoding_tensor.append(0)
    encoding_tensor[1]=pos
    version_mut={0:34406,2:41000,3:42500}
    encoding_tensor[0]=version_mut[version]+(pos-1)*5+target_nuc
    # bug area, needs fix
    if version==0:
        encoding_tensor[2]=29904+(current_nuc-1)*5+target_nuc
    if version==3:
        encoding_tensor[2]=29904+current_nuc*5+target_nuc

    annos_visible=['ORF1a','ORF1b','S','ORF3a','E','M','ORF6','ORF7a','ORF7b','ORF8','N','ORF10']
    annos_visible_dic={}
    annos_visible_start={}

    assigned_pos=0
    
    for i in range(len(annos_visible)):
        annos_visible_dic[annos_visible[i]]=i
        item=annos_visible[i]
        annos_visible_start[item]=assigned_pos
        anno_length=(anno[item]['end']-anno[item]['start']+1)/3
        assigned_pos+=anno_length
      
    annos_overlap=['ORF0','ORF3b','ORF3c','ORF3d','ORF3d-2','ORF9b','ORF9c']
    annos_overlap_dic={}
    annos_overlap_start={}
    for i in range(len(annos_overlap)):
        annos_overlap_dic[annos_overlap[i]]=i
        item=annos_overlap[i]
        annos_overlap_start[item]=assigned_pos
        anno_length=(anno[item]['end']-anno[item]['start']+1)/3
        assigned_pos+=anno_length

    trans_table_item=['K','N','T','R','S','I','Q','M','H','P','L','E','D','A','G','V','O','Y','C','W','F','-']
    aa_dic={}
    for i in range(len(trans_table_item)):    
        aa_dic[trans_table_item[i]]=i
      
    #print(anno)
    for w in anno:
        item=anno[w]
        if w in annos_visible:
            
            if item['start']<=pos and item['end']>=pos:
                aa_start=pos-(pos-item['start'])%3
                target_aa='_'
                if not('-' in new_ref[aa_start-1:aa_start+2]):
                    old_aa=trans_table[current_ref[aa_start-1:aa_start+2]]
                    new_aa=trans_table[new_ref[aa_start-1:aa_start+2]]
                    if old_aa!=new_aa:
                        target_aa=new_aa
                else:
                    if not('-' in current_ref[aa_start-1:aa_start+2]):
                        old_aa=trans_table[current_ref[aa_start-1:aa_start+2]]
                        target_aa='-'
                    
                if target_aa in aa_dic:
                    old_aa=aa_dic[old_aa]
                    target_aa=aa_dic[target_aa]
                    if version==0:
                        encoding_tensor[3]=29924+annos_visible_dic[w]
                        encoding_tensor[4]=29943+(aa_start-item['start'])//3
                        encoding_tensor[5]=old_aa*22+target_aa+33944
                    if version==2: #pos only
                        encoding_tensor[2]=29904+annos_visible_start[w]+(aa_start-item['start'])//3
                    if version==3:
                        encoding_tensor[3]=29924+annos_visible_start[w]+(aa_start-item['start'])//3
                        encoding_tensor[4]=42000+old_aa*22+target_aa
        has_overlap=0
        if w in annos_overlap:
            if item['start']<=pos and item['end']>=pos and has_overlap==0:
                aa_start=pos-(pos-item['start'])%3
                target_aa='_'
                if not('-' in new_ref[aa_start-1:aa_start+2]):
                    old_aa=trans_table[current_ref[aa_start-1:aa_start+2]]
                    new_aa=trans_table[new_ref[aa_start-1:aa_start+2]]
                    if old_aa!=new_aa:
                        target_aa=new_aa
                else:
                    if not('-' in current_ref[aa_start-1:aa_start+2]):
                        old_aa=trans_table[current_ref[aa_start-1:aa_start+2]]
                        target_aa='-'
                    
                if target_aa in aa_dic:
                    has_overlap=1
                    old_aa=aa_dic[old_aa]
                    target_aa=aa_dic[target_aa]
                    if version==0:
                        encoding_tensor[6]=29936+annos_overlap_dic[w]
                        encoding_tensor[7]=29943+(aa_start-item['start'])//3
                        encoding_tensor[8]=old_aa*22+target_aa+33944
                    if version==2: #pos only
                        encoding_tensor[3]=29904+annos_overlap_start[w]+(aa_start-item['start'])//3
                    if version==3:
                        encoding_tensor[5]=29924+annos_overlap_start[w]+(aa_start-item['start'])//3
                        encoding_tensor[6]=42000+old_aa*22+target_aa
    return encoding_tensor,new_ref

def read_country_list():
    replace_keys={'changzhou':'jiangsu','guangzhou':'guangdong','hangzhou':'zhejiang','pingxiang':'jiangxi',
              'shangrao':'jiangxi','shaoxing':'zhejiang','weifang':'shandong','yingtan':'jiangxi',
            'harbin':'heilongjiang','jian':'jiangxi','jiujiang':'jiangxi','changde':'hunan',
            'lishui':'zhejiang','foshan':'guangdong','jining':'shandong','xinyu':'jiangxi','nanchang':'jiangxi',
            'fuzhou':'fujian','yichun':'jiangxi','tianmen':'hubei','kashgar':'xinjiang',
            'cotedivoirecotedivoire':'cotedivoire','chinay':'china','brasil':'brazil','mexicomex':'mexico',
            'urumqi':'xinjiang','luan':'anhui','chilema':'chile','shulan':'jilin','taly':'italy',
            'cotedivoireafrica':'cotedivoire','gd':'guangdong','tianjn':'tianjin','ialy':'italy','spaiin':'spain',
            'fance':'france','romnaia':'romania','lka':'srilanka','wuhan':'hubei','shenzhen':'guangdong',
            'jingzhou':'hubei','ganzhou':'jiangxi','mauritanie':'mauritania','cameroun':'cameroon','us':'usa',
            'spainandplub':'saotomeandprincipe','mex':'mexico','qingdao':'shandong','saudi':'saudiarabia',
            'botswna':'botswana','dji':'djibouti','pdl':'portugal','zambai':'zambia','saintmartin':'saintmarten',
            'congo':'republicofthecongo','africa':'southafrica','dom':'dominicanrepublic','drcongo':'democraticrepublicofthecongo',
            'andorre':'andorra','afganistan':'afghanistan','unitedkingom':'unitedkingdom','guyane':'guyana',
            'tahiti':'frenchpolynesia','westbank':'palestine','macedonia':'northmacedonia','antigua':'antiguaandbarbuda',
            'austraila':'australia','erbil':'iraq','macau':'macao','easttimor':'timorleste','turksandcaicos':'turksandcaicosislands',
            'italia':'italy','newcaledonie':'newcaledonia','tibet':'xizang','argentino':'argentina','brazi':'brazil',
            'spin':'spain','mexicoi':'mexico','wallisetfutuna':"newcaledonia",'china//guangxi':'guangxi','jersey':'england'
            }

    w=open("country_population.txt",'r')
    lines=w.readlines()
    country_list={}
    nowcountries=0
    for l in lines:
        lsp=l.split('\t')
       # print(lsp)
        if len(lsp)<2:
            continue
        od=ord(lsp[0][0])
        if (od>=48 and od<=57):
            od=ord(lsp[1][0])
            if (od>=48 and od<=57):
                country_name=lsp[2]
                country_pop=lsp[3]
            else:
                country_name=lsp[1]
                country_pop=lsp[2]
        else:
            country_name=lsp[0].split('[')[0]
            country_pop=lsp[1]
        country_name=country_name.replace(' ','').replace('&','and').lower()
        if country_name in replace_keys:
            country_name=replace_keys[country_name]
        if not(country_name in country_list):
            country_list[country_name]=nowcountries
            nowcountries+=1
    return country_list


def location_time_encoding(region, time,country_list,version=0):
    # encoding: 0: nothing
    # 183921-184286   country/region
    # 184300-184310   time(2019-2025)
    # 184311-184394   time (2019.1--2025.12)
    # 184401-184531   date
    version_base_dic={0:183921,1:149600,2:190600, 3:192100}
    version_base=version_base_dic[version]
    region_keys={'jiangsu':'china','guangdong':'china','zhejiang':'china','jiangxi':'china',
             'shandong':'china','hunan':'china','anhui':'china','fujian':'china','henan':'china',
             'tianjin':'china','beijing':'china','shanghai':'china','sichuan':'china','chongqing':'china',
             'england':'unitedkingdom','scotland':'unitedkingdom','yunnan':'china','guangxi':'china',
            'fujian':'china','puertorico':'usa','guam':'usa','gansu':'china','usvirginislands':'usa',
            'wales':'unitedkingdom','northernireland':'unitedkingdom','shanxi':'china','shaanxi':'china',
            'innermongolia':'china','americansamoa':'usa','hubei':'china','liaoning':'china','hainan':'china',
            'hebei':'china','jilin':'china','guizhou':'china','xinjiang':'china','qinghai':'china',
            'northernmarianaislands':'usa','usasouthdakota':'usa','chongqing':'china','ningxia':'china',
            'hongkong':'china','canaryislands':'spain','crimea':'russia','bonaire':'netherlands','macao':'china',
            'montserra':'unitedkingdom','inteustatius':'netherlands','xizang':'china','heilongjiang':'china',
            'ladakh':'india'
            }
    encoding_tensor=[0,0,0,0,0]
    if '+' in region:
        country,region=region.split('+')
    else:
        country=region
        if country in region_keys:
            country=region_keys[region]

    if country in country_list:
        encoding_tensor[0]=country_list[country]+version_base
    if region in country_list:
        encoding_tensor[1]=country_list[region]+version_base
    dy=time.split('-')
    if dy[0] in ['2019','2020','2021','2022','2023','2024','2025']:
        yy_enc=int(dy[0])-2019
        encoding_tensor[2]=yy_enc+version_base+379
    if len(dy)>1:
        try:
            mm=int(dy[1])
            yymm_enc=yy_enc*12+mm
            encoding_tensor[3]=yymm_enc+version_base+390
        except:
            print(time)
    if len(dy)==3:
        try:
            mm=int(dy[2])
            encoding_tensor[4]=mm+version_base+480
        except:
            print(time)
    return encoding_tensor

# shuffle data based on log of weight
# new weight=log(weight/100)
def decode_mutations(mutation,version=0):
    # encoding criteria: 
    # 0: nothing 
    # pos: 1-29903 
    # 29904-29923: A->T  A->C  A->G  T->A T->C  T->G  C->A C->T C->G G->A G->T G->C A->- T->- C->- G->-
    # 29924-29935 : annos   (Orf1a, Orf1b, S, Orf3a, E,M, Orf6, Orf7a, Orf7b, Orf8, N, Orf10 )
    # 29936-29942 : overlap_annos (Orf0, Orf3b, Orf3c, Orf3d, Orf3d-2, Orf9b, Orf9c)
    # 29943-33943 : position of related 1-4401
    # 33944-34405 : 21*22=462 transition map
    # 34406-183920 : 29903*5=149515 mutation profiles
    # mutation is in form of X0000Y 

    pos=(mutation[0]-34406)//5
    target=(mutation[0]-34406)%5
    
    nuc_dic={'A':0,"T":1,"C":2,'G':3,'-':4}
    mutation0=str(pos)+nuc_dic[target]

    pos_int=mutation[1]
    init_mut_transform=nuc_dic[(mutation[2]-29904)//5]
    result_mut_transform=nuc_dic[(mutation[2]-29904)%5]
    mutation1=init_mut_transform+str(pos_int)+result_mut_transform

    annos_visible=['ORF1a','ORF1b','S','ORF3a','E','M','ORF6','ORF7a','ORF7b','ORF8','N','ORF10']
    annos_overlap=['ORF0','ORF3b','ORF3c','ORF3d','ORF3d-2','ORF9b','ORF9c']

    trans_table_item=['K','N','T','R','S','I','Q','M','H','P','L','E','D','A','G','V','O','Y','C','W','F','-']
    aa_dic={}
    for i in range(len(trans_table_item)):    
        aa_dic[trans_table_item[i]]=i

        

    return 0


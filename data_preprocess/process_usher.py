import json
import argparse

parser = argparse.ArgumentParser(description='Demo of argparse')
parser.add_argument('--date', type=str,default='2025-07-16')
parser.add_argument('--input-tree', type=str,default=None)
args = parser.parse_args()
input_tree=args.input_tree
data_date=args.date

yy,mm,dd=data_date.split('-')
# build a dic by country and date
country_date_dic={}
country_pop_dic={}

country_heading_dic={}
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
removed_keys=['aus','chn','egy','fra','irq','gbr','pak','ita','fin','nzl','deu','bra','ken','irn','drc','isr',
              'mmr','jpn','rna','swe','geo','bgd','dnk','rus','grc','jor','esp','per','gha','nga','bel','kor',
              'tha','col','mex','ury','sau','tur','gmb','uzb','uga','ven','gtm','unitedstates','chl','srb',
              'tun','zaf','mvi','che','arg','lby','prt','can','mrt','cmr','mli','gab','mar','aut','phl','nrd',
              'irl','hu','twn','lbn','blr','pri','notprovided','noseandthroatswab','eth','bhr','oropharyngealswab',
              'idn','zwec','ben','cze','dom','vnm','h.sapiens','rou','pdl','nor','mng','notidentified',
              'homosapeins','lao','kaz','sgp','ecu','pol','arm','mda','pry','sur','lka','sle','vero','dji','som',
              'svk','mlt','verocells','civ','homo','qat','syc','kwt','virusstock','hun','northerngreatergalago',
              'sagamiharajpn','tokyojpn','human','hkg','homospiens','nld','mys','tls','virus']
removed_strs=['norw']
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
            'spin':'spain','mexicoi':'mexico','wallisetfutuna':"newcaledonia",'china//guangxi':'guangxi','jersey':'england',
            'korea':'southkorea'
            }

state_keys={'czb':'ca','search':'ca','mdh':'mn','uf':'fl','nmdoh':'nm','ufh':'fl','vi':'va','altcov':'wa','cumc':'ne','nminbre':'nm',
            'und':'in','gu':'guam','y':'co','insphl':'in','cd':'hi','icu':'nj','cuimc':'ny','mamc':'wa','mp':'northernmarianaislands',
           'nn':'nm','txelp':'tx','sphl':'tn','cmi':'oh','as':'co','pn':'pa','nys':'ny','wausafsam':'wa','flusafsam':'fl',
           'mtusafsam':'mt','view':'tn','usafl':'fl'
           }
india_keys={'leh':'ladakh','pun':'pb','drde':'mp','dh':'dn','nd':'delhi',
           'mcw':'delhi','ke':'kl','klmcl':'kl','klkz':'kl','mg':'ml','bh':'delhi','na':'cg'}

vanish_keys=['cat','env','mesocricetusauratus','mouse','bird','rodent','mink','dog','lion','deer',
             'neovisonvison','canisfamiliaris','gorilla','tiger','ferret','snowleopard',
             'mink','musmusculus','puma','canine','asiaticlion','otter','caprahircus','ovisaries',
             'susscrofa','canislupus','hippo','hamster','odocoileusvirginianus','hyena','lacertilia',
             'bostaurus','caoti','monkey','armadillo','leopard','syrianhamster','dogdomestic','country',
             'catdomestic','minkamerican','whitetaileddeer']

def process_shortlists():
    us_shortlist={}
    f=open("us_state_shortlist.txt",'r')
    for line in f:
        w=line.split('\t')
        simplified=w[-1].lower().replace('\n','')
        full=w[0].lower().replace(' ','').replace('\n','')
        us_shortlist[simplified]=full
        region_keys[full]='usa'
    india_shortlist={}
    f.close()
    f=open("india_state_shortlist.txt")
    for line in f:
        w=line.split('\t')
        simplified=w[-1].lower().replace('in-','').replace('\n','')
        simplified=simplified.split(';')
        full=w[0].lower().replace(' ','').replace('\n','')
        for item in simplified:
            india_shortlist[item]=full
        region_keys[full]='india'
    f.close()
    return us_shortlist,india_shortlist

def process_str(st):
    st=st.replace(' ','')
    st=st.replace('_','')
    st=st.replace("'",'')
    st=st.replace('-','')
    st=st.lower()
    if st in replace_keys:
        st=replace_keys[st]
    if st in vanish_keys:
        return None
    
    return st

def read_country_pop():
    w=open("country_population.txt",'r')
    lines=w.readlines()
    pop_dic={}
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
        country_pop=int(country_pop.replace(',',''))
        pop_dic[country_name]=country_pop
    return pop_dic

def process_usher(filename):
    seqcountdic={}
    nums=0
    presented_ctr=[]
    f=open(filename,'r')
    lines=f.readlines()
    l0=json.loads(lines[0])
    mutations=l0['mutations']
    countries={}
    existing_bad=[]
    outlines=open("usher_tree"+data_date+".txt",'w')
    print(lines[0],file=outlines)

    for item in lines[1:]:
        vi=json.loads(item)
        if vi["meta_country"]!="":
            country=vi["meta_country"]
            country=process_str(country)
            heading=""

            first=vi['name'].split('|')[0]
            if 'USA'==first[0:3] or 'India'==first[0:5]:
                first=first.replace('_','-')
                heading=first.split('-')[0]
            else:
                heading=first.split('/')[0]
            
            heading=heading.lower().replace('.','').replace(' ','')
            if 'IMS' in heading:
                heading=''
            if len(heading)>0:
                for ind in range(48,58):
                    if chr(ind) in heading:
                        heading=''
            heading=process_str(heading)
            if country=='?':
                country=heading.split('/')[0]
            
            if not(country in pop_dic) and not(country in presented_ctr):
                print(country,heading)
                presented_ctr.append(country)
        
            if heading is not None:
                if heading=='' or heading in removed_keys:
                    heading=country
                for keys in removed_strs:
                    if keys in heading:
                        heading=country
                if '/' in heading:
                    if 'usa' in heading:
                        country='usa'
                        heading=heading.split('/')[-1]
                        if heading in state_keys:
                            heading=state_keys[heading]
                        if heading in us_shortlist:
                            heading=us_shortlist[heading]
                        if not(heading in pop_dic):
                            heading='usa'
                    if 'india' in heading:
                        country='india'
                        heading=heading.split('/')[-1]
                        if heading in india_keys:
                            heading=india_keys[heading]
                        if heading in india_shortlist:
                            heading=india_shortlist[heading]
                        if not(heading in pop_dic):
                            heading='india'

                if country in region_keys:
                    country=region_keys[country]
                
                
                if not(heading in pop_dic):
                    heading=country
                
                if not(country in pop_dic):
                    print(vi,country,"type 1")
                if (country!=heading and not(heading in region_keys)):
                    #print(vi,country,heading,'type 2')
                    country=heading

                cheading=country+'+'+heading
                date=vi['meta_date']
                datesp=date.split('-')
                yy=datesp[0]
                
                if (len(datesp)>1) :
                    mm=datesp[1]
                    if len(mm)==1:
                        mm='0'+mm
                    yymm=yy+mm
                    if not(cheading in seqcountdic):
                        seqcountdic[cheading]={}
                    if not(yymm in seqcountdic[cheading]):
                        seqcountdic[cheading][yymm]=0
                    seqcountdic[cheading][yymm]+=1
                    vi['yymm']=yymm
                vi['location']=cheading
                
                

                if not(cheading in country_heading_dic):
                    country_heading_dic[cheading]=1
                    #print(cheading,vi['name'])
                else:
                    country_heading_dic[cheading]+=1
                nums+=1
                if nums%1000000==0:
                    print(nums)
                print(json.dumps(vi),file=outlines)    
        else:
            print(json.dumps(vi),file=outlines)       
    outlines.close()
    #print(seqcountdic)
    return seqcountdic

def proc(num):
    import numpy as np
    if num<100000:
        return num
    if num>10000000:
        return 1000000
    return 100000*np.sqrt(num/100000)

def compute_weight(seqcountdic):
    import copy
    sumdic=copy.deepcopy(seqcountdic)
    weightdic={}
    for item in sumdic:
        country,region=item.split('+')
        if country!=region:
            ctr=country+'+'+country
            if ctr in sumdic:
                for yymm in sumdic[item]:
                    if yymm in sumdic[ctr]:
                        sumdic[item][yymm]+=seqcountdic[ctr][yymm]*pop_dic[region]/pop_dic[country]
                        sumdic[ctr][yymm]+=seqcountdic[item][yymm]
    for item in sumdic:
        country,region=item.split('+')
        if not(item in weightdic):
            weightdic[item]={}
        for yymm in sumdic[item]:
            weightdic[item][yymm]=pop_dic[region]/sumdic[item][yymm]
            weightdic[item][yymm]=proc(weightdic[item][yymm])
            
    return weightdic
            



pop_dic=read_country_pop()
#print(pop_dic)
us_shortlist,india_shortlist=process_shortlists()
#print(us_shortlist,india_shortlist)
if input_tree is None:
    seqcountdic=process_usher("../gisaidAndPublic."+data_date+".masked.gisaidNames.taxonium.jsonl")
else:
    seqcountdic=process_usher(input_tree)
weightdic=compute_weight(seqcountdic)

json.dump(weightdic,open('weight_dic_proc_'+mm+dd+'.json','w'))


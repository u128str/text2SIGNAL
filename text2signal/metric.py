import pandas as pd
from deepdiff import DeepDiff
from difflib import SequenceMatcher as SM
import json
import yaml

# https://www.sbert.net/docs/pretrained_models.html#model-overview
from sentence_transformers import SentenceTransformer, util

#model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1') 


def dict_generator(indict, pre=None):
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value,  pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for k,v in enumerate(value):
                    for d in dict_generator(v, pre + [key] + [k]):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]

def compute_score(gt,pr,model_name='multi-qa-MiniLM-L6-cos-v1'):
    """
    """
    model = SentenceTransformer(model_name) 
    
    linerized=[]
    for e in  dict_generator(gt):
        e.append(1.0)
        e.append(1.0)
        linerized.append(e)
        
    #gt=gt1
    #pr=pr1
    deep_diff=DeepDiff(gt,pr, ignore_order=False, cache_size=5000, get_deep_distance=True,ignore_private_variables=False) #, view="tree")
    log=[]
    for el in linerized:
        form_compatible_key="root"+str(el[:-3]).replace(", ","][")  #  "root['work_experiences'][0]['industry']
        if 'values_changed' in deep_diff.to_dict().keys():
            if form_compatible_key in deep_diff.to_dict()['values_changed']:
                df_to_anls=deep_diff.to_dict()['values_changed'][form_compatible_key]
                new_value=df_to_anls['new_value']
                old_value=df_to_anls['old_value']
                sm_simple=0.0
                sm=0.0
                if type(new_value) == str and type(old_value) == str:
                    sm_simple =  SM(None, new_value, old_value ).ratio() 
                    ##sm = util.dot_score( model.encode(el.t1),  model.encode(el.t2))
                    sm= util.cos_sim(model.encode(new_value),  model.encode(old_value)).item()
                el[-2]=sm
                el[-1]=sm_simple
                print(el[:-3],"|", old_value , "|<-gt pr->|", new_value, " | Similarities Embd/Simple:", sm, sm_simple )
                #log.append("{} |  {} |<-gt pr->| {} | Similarities Embd/Simple: {}/{} ".format(el[:-3],old_value,new_value,sm, sm_simple))


    
    # linerized_unique
    linerized_unique=[]
    for el in linerized:
        a = ".".join([x for x in el[:-3] if not isinstance(x, int)]) # remove int from full item address 'skills', 0, 'context' - > 'skills.context'
        linerized_unique.append([a]+el[-2:])
    #linerized_unique
    #print(linerized_unique)
    
    #Embd Score
    df = pd.DataFrame(linerized_unique, columns = ['Name', 'Score_Embd', "Score"])
    st1 = df.groupby(['Name'])[ 'Score_Embd'].describe()[['mean', 'count']].reset_index() 
    st12 = df[df.Score_Embd == 1.0].groupby(['Name']).size().reset_index().rename(columns={0:'match_Embd'})
    res1 = pd.merge(st1, st12, how="left", on=["Name"]).fillna(0)
    res1.rename(columns={'mean':'Avr. Score_Embd','count':'Support_Embd'}, inplace=True)
    res1.set_index('Name', inplace=True)
    
    # Simple Score
    st2 = df.groupby(['Name'])['Score'].describe()[['mean', 'count']].reset_index() 
    st21 = df[df.Score == 1.0].groupby(['Name']).size().reset_index().rename(columns={0:'match_Simple'})
    res2 = pd.merge(st2, st21, how="left", on=["Name"]).fillna(0)
    res2.rename(columns={'mean':'Avr. Score_Simple','count':'Support_Simple'}, inplace=True)
    res2.set_index('Name', inplace=True)
    
    summary = pd.merge(res1,res2,how="left", on=["Name"])
    
    list_of_lists = []
    list_of_lists.append(["Embd.",res1['Avr. Score_Embd'].mean(), res1['Support_Embd'].sum(), res1['match_Embd'].sum()])
    list_of_lists.append(["Simple",res2['Avr. Score_Simple'].mean(), res2['Support_Simple'].sum(), res2['match_Simple'].sum()])
    
    total_emb_json= util.cos_sim(model.encode(json.dumps(gt)),model.encode(json.dumps(pr))).item()
    total_emb_yaml= util.cos_sim(model.encode(yaml.dump(gt)),model.encode(yaml.dump(pr))).item()
    
    
    list_of_lists.append(["Embd JSON",total_emb_json])
    list_of_lists.append(["Embd YAML",total_emb_yaml])
    total= pd.DataFrame(list_of_lists, columns=['Method','Score', 'Support' , 'match']).reindex().fillna("")

    
    return total, summary #, log
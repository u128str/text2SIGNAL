import argparse
import json
import warnings
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fine_tuning.training.utils import get_tokenizer

warnings.filterwarnings('ignore', 'Setting `pad_token_id` to `eos_token_id`', UserWarning)


def get_completions(model, tokenizer, prompts, max_tokens=500):
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
    #inputs = {key: value.to('cuda') for key, value in inputs.items()}
    tokens = model.generate(**inputs, max_new_tokens=max_tokens, penalty_alpha=0.6, top_k=4)
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def get_ll(model, tokenizer, sequence):
    # Tokenize the sequence and obtain input_ids
    input_ids = tokenizer(sequence, return_tensors='pt')['input_ids']

    # Obtain the model's logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Shift the input_ids to the left to obtain the target tokens
    target_ids = input_ids[:, 1:].contiguous()

    # Reshape logits to (-1, vocab_size) and target_ids to (-1,)
    logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    target_ids = target_ids.view(-1)

    # Compute the log likelihood
    log_probs = F.log_softmax(logits, dim=-1)
    log_likelihood = log_probs[range(target_ids.size(0)), target_ids].sum().item()

    return log_likelihood


def write_ll(filename, log_likelihoods):
    with open(filename, 'w') as file:
        for log_likelihood in log_likelihoods:
            file.write(str(log_likelihood) + '\n')


def load_json(file_path):
    with open(file_path, 'r') as f:
        #obj = json.load(f)
        obj = [json.loads(line) for line in f]
    return obj


def write_json(file_path, obj):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--gt_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--is_deepspeed', action='store_true')
    parser.add_argument('--get_ll', action='store_true')
    parser.add_argument('--llm_name',type=str)

    return parser.parse_args()

def signavio_prompt(description_llm):
    """
    """
    #import yaml

    mistral="""
  <s>[INST]
  You are SIGNAL assistant, a part of SAP Signavio's Process Intelligence Suite. 
  SIGNAL stands for Signavio Analytics Language. 
  SIGNAL is a dialect of SQL.
  Your goal is to help users craft SIGNAL 
  queries and understand the SIGNAL language better. 
  
  
  ### Instruction:
  
  Task to solve:
  
  Construct SIGNAL expression querying given textual description in Input.
  Expected output results are in JSON format.
  
  
  ### Input:
  {description_llm}
  
  ### Output:
  [/INST]
"""
    #mistral_json=yaml.load(mistral,  Loader=yaml.FullLoader)
    #return mistral_json["query_template"].format(description_llm=str(description_llm))
    return mistral.format(description_llm=description_llm) 

def parser(txt):
    #res = txt.split("[/INST]")[-1]
    res=txt.replace('\"query\":','"query":')
    res=res.replace("\'query\':",'"query":')
    res=res.replace("'query':",'"query":')
    res="".join(res.split('"query":')[-1].split('}')[:-1])
    return res

def main():
    
    args = get_args()
    gt = load_json(args.gt_path)
    tokenizer = get_tokenizer(args.model_id)

    #prompts = [' '.join(gt_snippet.split()) for gt_snippet in gt]  #['SAP is', 'SAP CEO', 'SAP Concur']
    
    #prompts = [ signavio_prompt(gt_snippet["description_llm"]) for gt_snippet in gt][0:3]
    prompts = gt #[0:101]

    llm_name="llmname"
    if args.llm_name:
        llm_name=str(args.llm_name)
        
    if args.is_deepspeed:
        from collections import OrderedDict

        state_dict = torch.load(args.model_path)

        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name = key[len('_forward_module.model.'):] if key.startswith('_forward_module.model.') else key
            new_state_dict[name] = value

        model = AutoModelForCausalLM.from_pretrained(args.model_id, device_map='auto', torch_dtype=torch.float16)
        # AutoModelForCausalLM.from_pretrained
        model.config.use_cache = True
        model.load_state_dict(new_state_dict)

    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                     device_map='auto',
                                                     use_cache=True,
                                                     torch_dtype=torch.float16)

    if args.get_ll:
        ll = []
        for sentences in gt:
            ll.append(get_ll(model, tokenizer, sentences))
        write_ll(args.output_path, ll)
    else:
        completions = []
        nfails=0
        #for prompt in prompts:
        nchnk=10
        start0 = time.time()
        for i in range(0, len(prompts), nchnk):
            start = time.time()  
            chunk = prompts[i:i + nchnk]
            #print(f'prompt is "{prompt}"\n\n')
            #llm_input= signavio_prompt(prompt["description_llm"])
            #completion = get_completions(model, tokenizer, [llm_input])[0]
            
            llm_input_chnk=[signavio_prompt(el["description_llm"]) for el in chunk]
            completion = get_completions(model, tokenizer, llm_input_chnk) #[0]
            for r, e in  zip (completion, chunk):
                e[f"llm_{llm_name}_full_output"]=r
                out =  str(parser(r))
                e[f"llm_{llm_name}_query"] = out
                if out == "":
                    nfails+=1
                
            completions=completions+chunk
            ### res = completion.split("[/INST]")[-1]
            ### res=res.replace('\"query\":','"query":')
            ### res="".join(completion.split('"query":')[-1].split('}')[:-1])
            ### if res == "":
            ###     nfails+=1
            ### #print("Completion: \n",{"llm_mistral_query":f'"""{res}"""'})
            ### #out= {"llm_mistral_query":str(res), "gt_query": prompt["query"], "full_output":completion}
            ### prompt["llm_mistral_query"] = str(res)
            ### prompt["llm_mistral_full_output"]=completion
            ### completions.append(prompt)
            print(i+nchnk, "out of ", len(prompts),  " remains: ",(len(prompts) - i - nchnk) )
            print( f"LLM {llm_name} inference took", time.time()- start0, " takes [m]:", (len(prompts) - i - nchnk) * (time.time()- start0) / (i+nchnk)/ 60 ) 
           
            

        write_json(args.output_path, completions)
        print(f'Problematic cases: {nfails} out of {len(completions)}')


if __name__ == '__main__':
    main()

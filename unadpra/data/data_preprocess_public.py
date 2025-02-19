
from llm2vec import LLM2Vec
hf_token = "hf_"
import os
import json
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
sys.path.append("..")
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXConfig, GPTNeoXModel, GPTNeoXPreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Literal, Optional
import nltk

# from transformers import LlamaTokenizerFast
# tokenizer=LlamaTokenizerFast.from_pretrained('/home/yanghongyu/llama3-8B')
def reward_handler_pro(r_path,stage):
    w_path=r_path.replace('_pro.json','_re.json')
    fw=open(w_path,'w')
    model_name = "/home/yanghongyu/pro/train_pro/reward-model-deberta-v3-large"
    # model_device = "cuda:{}".format(rank)
    model_device=torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "right"
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
    reward_model.eval()

    lines=open(r_path,'r').readlines()
    for line in tqdm(lines):
        line=json.loads(line.strip())
        prefixes=line['prefix']
        suffixes=line['suffix']
        input_content = tokenizer(
            prefixes,
            suffixes,
            padding=True,
            truncation=True,
            max_length=512, #1024
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            rewards = reward_model(**input_content).logits
        line["pro_reward"]=rewards.view(-1).detach().cpu().numpy().tolist()
        fw.write(json.dumps(line)+'\n')

def reward_handler(r_path):
    device=torch.device("cuda")
    w_path=r_path.replace(".json","_se.json")
    l2v = LLM2Vec.from_pretrained(
        "/home/yanghongyu/codeqa/llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp",
         peft_model_name_or_path="/home/yanghongyu/codeqa/llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        # device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        device_map="cuda" ,
  
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)

    instruction = (
    "Given an question, please answer it:")
    lines=open(r_path,'r').readlines()
    fw=open(w_path,'w')
    err_cnt=0
    for line in tqdm(lines):
        line=json.loads(line.strip())
        qid=line['qid']
        question=line['question']
        answers=line['answers']
        queries = [
            [instruction,question]]
        answers=line['answers']
        answer_pools=[]
        for index,a in enumerate(answers):
            answer_pools.append(a["text"])
            a['answer_id']=index
        # Compute cosine similarity
        q_reps = l2v.encode(queries)
        a_reps = l2v.encode(answer_pools)
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        a_reps_norm = torch.nn.functional.normalize(a_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, a_reps_norm.transpose(0, 1)) 
        # add check the accepted answer similarity whether the most 
        line["sim_scores"]=cos_sim[0].tolist()
    
        fw.write(json.dumps({qid:line})+'\n')
    
    print(w_path,err_cnt)  

def generate_pro_data_sa(r_path,stage):
    lines=open(r_path,'r').readlines()
    print(len(lines))
    w_path=r_path.replace("_se.json","_pro.json")
    fw=open(w_path,'w')
    stage=int(stage)
    sft_cnt=0
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post=list(line.values())[0]
        data={}
       
        question=post["question"]
        query='[ANSWER THE QUERY IN THE CODE COMMUNITY]'+'\n[QUERY]:'+question+"\n[END]:"
        data['prefix']=[query]*stage
        data['suffix']=[answer["text"] for answer in post["answers"][:stage]]
        for answer in post["answers"]:
            if answer["selected"]:
                data['sft_index']=answer["answer_id"]
                if data['sft_index']>=stage:
                    sft_cnt+=1

                    break
                break
        data["reward"]=post["sim_scores"][:stage]
        data["vote"]=[answer["pm_score"] for answer in post["answers"][:stage]]
       
        data["query_time"]=post["date"]
        fw.write(json.dumps(data)+'\n')
    print("sft index err: ",sft_cnt)
import os
import json
import random
from sklearn.model_selection import train_test_split



def split_data(data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    train_data, temp_data = train_test_split(data, test_size=(1 - train_ratio), random_state=42)
    valid_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (valid_ratio + test_ratio)), random_state=42)
    return train_data, valid_data, test_data

def write_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as fw:
        for d in tqdm(data):
            fw.write(json.dumps(d))
            fw.write('\n')


def combine_split_dataset1(r_path,w_dir):
    all_data = []
    lines=open(r_path,'r').readlines()
    for line in tqdm(lines):
        line=json.loads(line.strip())   
        all_data.append(line)
    print(len(all_data))
    train_data, valid_data, test_data = split_data(all_data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
    os.mkdir(f'{w_dir}/train')
    os.mkdir(f'{w_dir}/dev')
    os.mkdir(f'{w_dir}/test')
    write_json_file(train_data, f'{w_dir}/train/train.json')
    write_json_file(valid_data, f'{w_dir}/dev/valid.json')
    write_json_file(test_data,f'{w_dir}/test/test.json')



def procedure(r_path,cnt):
    reward_handler(r_path)
    r_path1=r_path.replace(".json","_se.json")
    generate_pro_data_sa(r_path1,cnt)
    r_path2=r_path1.replace("_se.json","_pro.json")
    reward_handler_pro(r_path2,cnt)
    r_path3=r_path2.replace("_pro.json","_re.json")
    dir='/'.join(r_path.split("/")[:8])
    combine_split_dataset1(r_path3,dir)


# deal che dataset
# reward_handler("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/train_selected_3.json")
# generate_pro_data_sa("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/train_selected_3_se.json",3)
# reward_handler_pro("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/train_selected_3_pro.json",3)
# combine_split_dataset1("/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com/train_selected_3_re.json","/home/yanghongyu/codeqa/data/public/data/chemistry.stackexchange.com") #19040,1638

#deal code dataset
# reward_handler("/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com/train_selected_5.json")
# generate_pro_data_sa("/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com/train_selected_5_se.json",5)
# reward_handler_pro("/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com/train_selected_5_pro.json",3)
# combine_split_dataset1("/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com/train_selected_5_re.json","/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com") #

#del cooking dataset
# procedure("/home/yanghongyu/codeqa/data/public/data/cooking.stackexchange.com/train_selected_5.json",5)


# procedure("/home/yanghongyu/codeqa/data/public/data/Stackoverflow.com/train_selected_5.json",5) 
# procedure("/home/yanghongyu/codeqa/data/public/data/electronics.stackexchange.com/train_selected_5.json",5)
# procedure("/home/yanghongyu/codeqa/data/public/data/gaming.stackexchange.com/train_selected_5.json",5) #
# procedure("/home/yanghongyu/codeqa/data/public/data/history.stackexchange.com/train_selected_3.json",3) #19040
# procedure("/home/yanghongyu/codeqa/data/public/data/math.stackexchange.com/train_selected_4.json",4) #6660
# procedure("/home/yanghongyu/codeqa/data/public/data/physics.stackexchange.com/train_selected_4.json",4)
# procedure("/home/yanghongyu/codeqa/data/public/data/politics.stackexchange.com/train_selected_3.json",)
# procedure("/home/yanghongyu/codeqa/data/public/data/law.stackexchange.com/train_selected_2.json",2)
# procedure("/home/yanghongyu/codeqa/data/public/data/security.stackexchange.com/train_selected_6.json",6)
# procedure("/home/yanghongyu/codeqa/data/public/data/travel.stackexchange.com/train_selected_3.json",3)
# procedure("/home/yanghongyu/codeqa/data/public/data/academia.stackexchange.com/train_selected_4.json",4)
procedure("/home/yanghongyu/codeqa/data/public/data/music.stackexchange.com/train_selected_4.json",4)
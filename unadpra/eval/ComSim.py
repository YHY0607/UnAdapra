import os
import json
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from llm2vec import LLM2Vec
hf_token = "hf_"


def reward_handler(r_path,w_path,start_index=0,end_index=64238):
    device=torch.device("cuda:1")
    l2v = LLM2Vec.from_pretrained(
        "llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp",
         peft_model_name_or_path="llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        # device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        device_map="cuda:1" ,
  
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)

    instruction = (
    "Given an answer that may contain code, retrieve the text most relevant to it based on the semantics and structure of the code:")
    lines=open(r_path,'r').readlines()
    fw=open(w_path,'w')
    err_cnt=0
    for line in tqdm(lines[start_index:end_index]):
        line=json.loads(line.strip())
        post_id=list(line.keys())[0]
        post=list(line.values())[0]
        title=post["title"]
        body=post["body"]
        # accepted_answer=post["answer_body"]
        queries = [
            [instruction,title+':'+body]]
        answer_pools=[]
        answers=post['answers']
        for a in answers:
            answer_pools.append(a["body"])
        # Compute cosine similarity
        q_reps = l2v.encode(queries)
        a_reps = l2v.encode(answer_pools)
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        a_reps_norm = torch.nn.functional.normalize(a_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, a_reps_norm.transpose(0, 1)) 
        # add check the accepted answer similarity whether the most 
        answer_id=post["answer_number"]
        max_index = torch.argmax(cos_sim)+1
        # print()
        if answer_id != max_index:
            err_cnt+=1
            continue
        # assert answer_id == max_index , f"The accepted answer is not the most similar, {answer_id}, {max_index}"
        # print(cos_sim)
        post["sim_scores"]=cos_sim[0].tolist()
    
        # Sort answers by similarity scores
        sorted_answers = sorted(answers, key=lambda x: cos_sim[0][answers.index(x)], reverse=True)
        post['answers'] = sorted_answers
    
        # Add similarity order to the post
        post['sim_orders'] = [answers.index(a) + 1 for a in sorted_answers]
    
        fw.write(json.dumps({post_id:post})+'\n')
    
    print(w_path,err_cnt)  
def reward_handler1(r_path,w_path,start_index=64238,end_index=666666666):
    device=torch.device("cuda")
    l2v = LLM2Vec.from_pretrained(
        "llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp",
         peft_model_name_or_path="llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        # device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        device_map="cuda" ,
  
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)

    instruction = (
    "Given a question that may contain code, retrieve the answer most relevant to it based on the semantics and structure of the code:")
    lines=open(r_path,'r').readlines()
    fw=open(w_path,'w')
    err_cnt=0
    for line in tqdm(lines[start_index:]):
        line=json.loads(line.strip())
        post_id=list(line.keys())[0]
        post=list(line.values())[0]
        title=post["title"]
        body=post["body"]
        # accepted_answer=post["answer_body"]
        queries = [
            [instruction,title+':'+body]]
        answer_pools=[]
        answers=post['answers']
        for a in answers:
            answer_pools.append(a["body"])
        # Compute cosine similarity
        q_reps = l2v.encode(queries)
        a_reps = l2v.encode(answer_pools)
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        a_reps_norm = torch.nn.functional.normalize(a_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, a_reps_norm.transpose(0, 1)) 
        # add check the accepted answer similarity whether the most 
        answer_id=post["answer_number"]
        max_index = torch.argmax(cos_sim)+1
        # print()
        # if answer_id != max_index:
        #     err_cnt+=1
        #     continue
        # assert answer_id == max_index , f"The accepted answer is not the most similar, {answer_id}, {max_index}"
        # print(cos_sim)
        post["sim_scores"]=cos_sim[0].tolist()
    
        # Sort answers by similarity scores
        # sorted_answers = sorted(answers, key=lambda x: cos_sim[0][answers.index(x)], reverse=True)
        # post['answers'] = sorted_answers
    
        # Add similarity order to the post
        # post['sim_orders'] = [answers.index(a) + 1 for a in sorted_answers]
    
        fw.write(json.dumps({post_id:post})+'\n')  
def generate_pro_data(r_path,w_path,stage):
    lines=open(r_path,'r').readlines()
    print(len(lines))
    fw=open(w_path,'w')
    stage=int(stage)
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post=list(line.values())[0]
        data={}
        query='[ANSWER THE QUERY IN THE CODE COMMUNITY]'+'\n[QUERY]:'+post["title"]+": "+post['body']+"\n[END]:"
        data['prefix']=[query]*stage
        data['suffix']=[answer["body"] for answer in post["answers"][:stage]]
        data['sft_index']=0 
        data["reward"]=post["sim_scores"][:stage]
        data["vote"]=[answer["score"] for answer in post["answers"][:stage]]
        data["time"]=[answer["time"] for answer in post["answers"][:stage]]
        data["query_time"]=post["time"]
        fw.write(json.dumps(data)+'\n')
def generate_pro_data1(r_path,w_path,stage):
    lines=open(r_path,'r').readlines()
    print(len(lines))
    fw=open(w_path,'w')
    stage=int(stage)
    for line in tqdm(lines):
        line=json.loads(line.strip())
        post=list(line.values())[0]
        data={}
        body=post["body"].split(' ')[:128]
        query='[ANSWER THE QUERY IN THE CODE COMMUNITY]'+'\n[QUERY]:'+post["title"]+": "+post['body']+"\n[END]:"
        data['prefix']=[query]*stage
        data['suffix']=[answer["body"] for answer in post["answers"][:stage]]
        data['sft_index']=0 
        fw.write(json.dumps(data)+'\n')

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

def combine_split_dataset(folder_path="../data/8-cnt"):
    all_data = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            lines=open(file_path,'r').readlines()
            for line in lines:
                line=json.loads(line.strip())   
                all_data.append(line)
    print(len(all_data))
    train_data, valid_data, test_data = split_data(all_data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
    write_json_file(train_data, f'../data/dataset/train.json')
    write_json_file(valid_data, f'../data/dataset/valid.json')
    write_json_file(test_data,f'../data/dataset/test.json')
def combine_split_dataset1(r_path):
    all_data = []
    lines=open(r_path,'r').readlines()
    for line in lines:
        line=json.loads(line.strip())   
        all_data.append(line)
    print(len(all_data))
    train_data, valid_data, test_data = split_data(all_data, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
    write_json_file(train_data, f'../data/dataset/2621/train/train.json')
    write_json_file(valid_data, f'../data/dataset/2621/dev/valid.json')
    write_json_file(test_data,f'../data/dataset/2621/test/test.json')

   
def combine_gpt(r_path1,r_path2,w_path):
    lines1=open(r_path1,'r').readlines()
    lines2=open(r_path2,'r').readlines()
    fw=open(w_path,'w')
    
    for l1,l2 in zip(lines1,lines2):
        new_rank=[]
        l1=json.loads(l1.strip())
        l2=json.loads(l2.strip())
        # print(l1['ranks'][0])
        ranks=l1['ranks'].split(',')
        # print(ranks)
        for rank in ranks:
            rank=rank.strip().lstrip('[').rstrip(']')
            rank=rank.lstrip('[')
            new_rank.append(int(rank))
        
        l2['ranks']=new_rank
        fw.write(json.dumps(l2)+'\n')
    fw.close()
# combine_gpt('../data/gpt4o_ranks.json','../data/gpt_five_pro.json','../data/gpt_gt.json')
def get_ranks(values):
        sorted_values = sorted(values, reverse=True)  # 降序排序
        ranking_dict = {value: idx for idx, value in enumerate(sorted_values, start=1)}
        rankings = [ranking_dict[value] for value in values]
        return rankings

def reward_handler_gpt(gpt_reference_path,other_infer_path):
    w_path=other_infer_path.replace("infer/infer_generate_","reward/") 
    device=torch.device("cuda")
    l2v = LLM2Vec.from_pretrained(
         "llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp",
         peft_model_name_or_path="llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        # device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        device_map="cuda" ,
  
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)

    instruction = (
    "Given an answer that may contain code, retrieve the text most relevant to it based on the semantics and structure of the code:")
    gpt_lines=open(gpt_reference_path,'r').readlines()
    infer_lines=open(other_infer_path,'r').readlines()
   
    fw=open(w_path,'w') 
    err_cnt=0
    for reference,infer in tqdm(zip(gpt_lines[:len(infer_lines)],infer_lines)):
        reference=json.loads(reference.strip())
        infer=json.loads(infer.strip())
        
        queries = [
            [instruction,infer["infer"]['t']]]
        # print(infer["infer"])
        answer_pools=[]
        for a in infer["suffix"]:
            answer_pools.append(a)
        # Compute cosine similarity
        q_reps = l2v.encode(queries)
        a_reps = l2v.encode(answer_pools)
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        a_reps_norm = torch.nn.functional.normalize(a_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, a_reps_norm.transpose(0, 1)).squeeze(0).numpy().tolist()
        sorted_pairs = sorted(enumerate(cos_sim), key=lambda x: x[1], reverse=True)
        sorted_indices = [index+1 for index, value in sorted_pairs]

        infer['cos_sim']=cos_sim
        infer["sim_ranks"]=sorted_indices
        infer['gpt_ranks']=reference['ranks']
        fw.write(json.dumps(infer)+'\n')
    fw.close()
def reward_handler_public(infer_path):
    w_path=infer_path.replace("cache","sim")
    device=torch.device("cuda")
    l2v = LLM2Vec.from_pretrained(
        "llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp",
         peft_model_name_or_path="llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        # device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        device_map="cuda" ,
  
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)

    instruction = (
    "Given an answer that may contain code, retrieve the text most relevant to it based on the semantics and structure of the code:")
    infer_lines=open(infer_path,'r').readlines()
   
    fw=open(w_path,'w') 
    err_cnt=0
    for infer in tqdm(infer_lines):
        infer=json.loads(infer.strip())
        
        queries = [
            [instruction,infer["infer"]['t']]]
        # print(infer["infer"])
        answer_pools=[]
        for a in infer["suffix"]:
            answer_pools.append(a)
        # Compute cosine similarity
        q_reps = l2v.encode(queries)
        a_reps = l2v.encode(answer_pools)
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        a_reps_norm = torch.nn.functional.normalize(a_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, a_reps_norm.transpose(0, 1)).squeeze(0).numpy().tolist()
        sorted_pairs = sorted(enumerate(cos_sim), key=lambda x: x[1], reverse=True)
        sorted_indices = [index+1 for index, value in sorted_pairs]

        infer['cos_sim']=cos_sim
        infer["sim_ranks"]=sorted_indices
        fw.write(json.dumps(infer)+'\n')
    fw.close()

def re_dir(directory):
    for filename in os.listdir(directory):
        
        filepath = os.path.join(directory, filename)
        reward_handler_gpt('./data/gpt_gt.json',
                  filepath)
re_dir("./inference_res/2621/infer")

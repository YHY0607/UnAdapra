import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import argparse
import json
from tqdm import tqdm
import evaluate
import re


def eval(file_name):
    print(file_name)
    rouge = evaluate.load('rouge')
    pattern = r'main_(.*?)_5_test'
    match = re.search(pattern, file_name)

    id = match.group(1)
    # read_path = os.path.join("./inference_res", "infer_main_{}_{}_{}".format(args.index, args.stage, file_name))
    # with open(file_name, 'r', encoding='utf-8') as f:
    infer_data =open(file_name,'r').readlines()
    bleu = 0
    avg_reward = 0
    avg_code_smi=0
    sim_reward=0
    predictions = []
    references = []
    hit_1,hit_3=0,0
    recall_1,recall_3=0.0,0.0
    eval_path= os.path.join("./inference_res/2621/result.json")
    fw=open(eval_path,'a',encoding='utf-8') 
    for line in tqdm(infer_data):
        line=json.loads(line.strip())
        sim_ranks=line["sim_ranks"]
        gpt_ranks=line["gpt_ranks"]
        # ndcgs.append(getNDCG(sim_ranks,gpt_ranks))


        if sim_ranks[0] in gpt_ranks[:1]:
            hit_1+=1
        if sim_ranks[0] in gpt_ranks[:2]:
            hit_3+=1
        intersection_1=set(sim_ranks[:1])&set(gpt_ranks[:1])
        recall_1+=len(intersection_1)/1
        intersection_3=set(sim_ranks[:3])&set(gpt_ranks[:3])
        recall_3+=len(intersection_3)/3
        # print(line["infer"])
        avg_reward += line['infer']['reward']
        avg_code_smi+=line['infer']['q_code_sim']
        sim_reward+=line['cos_sim'][int(line["gpt_ranks"][0]-1)]
        bleu += line['infer']['bleu']
        if 'llama3' in filename:
            predictions.append(
            line['infer']["t"]["content"].strip())
        else:
            predictions.append(
            line['infer']["t"].strip())
        references.append(
            line['suffix'][int(line["gpt_ranks"][0]-1)].strip()
        )
    
    results = rouge.compute(predictions=predictions, references=references)
    bleu = bleu / len(infer_data)
    avg_reward = avg_reward / len(infer_data)
    avg_code_smi=avg_code_smi/len(infer_data)
    hit_1=hit_1/len(infer_data)
    hit_3=hit_3/len(infer_data)
    recall_1=recall_1/len(infer_data)
    recall_3=recall_3/len(infer_data)
    sim_reward=sim_reward/len(infer_data)
    
    fw.write(json.dumps({id:
            {"hit_1":hit_1,
             "hit_3":hit_3,
             "recall_1":recall_1,
             "recall_3":recall_3,
             "reward1":avg_reward,
             "sim_reward":sim_reward,
             "code_smi":avg_code_smi,
             "bleu":bleu,
             'rouge-l':results['rougeL'],
            }})+'\n')
        
if __name__ == "__main__":
    directory = "./inference_res/2621/eval"
    for filename in os.listdir(directory):
        
        filepath = os.path.join(directory, filename)
        if filepath.endswith('.json'):
            eval(filepath)
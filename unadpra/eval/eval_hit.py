
import argparse
import json
import tqdm
import numpy as np
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--index', type=str)
    parser.add_argument('--stage', type=int)
    args = parser.parse_args()
    return args
def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)+1),
        # np.divide(scores, np.log2(np.arange(scores.shape[0], dtype=np.float32) + 2)+1),
        dtype=np.float32)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)
    # print(rank_scores)
    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def eval_hit_2621(r_path):
    w_path=r_path.replace('infer_generate_main_','hit/').replace('_5_test.json','_hit.json')
    lines=open(r_path).readlines()
    hit_1,hit_3=0,0
    recall_2,recall_4=0.0,0.0
    ndcgs=[]
    for line in lines:
        scores= json.loads(line)["cos_sim"]
        sim_ranks=json.loads(line)["sim_ranks"]
        gpt_ranks=json.loads(line)["gpt_ranks"]
        # ndcgs.append(getNDCG(sim_ranks,gpt_ranks))


        if sim_ranks[0] in gpt_ranks[:1]:
            hit_1+=1
        if sim_ranks[0] in gpt_ranks[:2]:
            hit_3+=1
        intersection_2=set(sim_ranks[:2])&set(gpt_ranks[:2])
        recall_2+=len(intersection_2)/2
        intersection_4=set(sim_ranks[:4])&set(gpt_ranks[:4])
        recall_4+=len(intersection_4)/4
    hit_1/=len(lines)
    hit_3/=len(lines)
    open(w_path,"w").write(f"hit_1:{hit_1}\nhit_3:{hit_3}\nrecall_2:{recall_2/len(lines)}\nrecall_4:{recall_4/len(lines)}")
def eval_hit_dir_2621(directory):
    for filename in os.listdir(directory):
        
        filepath = os.path.join(directory, filename)
        if filepath.endswith('.json'):
            cnt=filepath[-11]
            eval_hit_2621(filepath)






def eval_hit_public_3(r_path,cnt):
    w_path=r_path.replace('infer_generate_main_','hit/').replace('_'+str(cnt)+'_test.json','.txt')
    lines=open(r_path).readlines()
    hit_1,hit_3=0,0
    recall=0.0
    for line in lines:
        sim_ranks=json.loads(line)["sim_ranks"]
        sft_index=json.loads(line)["sft_index"]
        loc=sim_ranks.index(sft_index+1)
        if sft_index+1 in sim_ranks[:1]:
            hit_1+=1
        recall+=1/(loc+1)
    
    hit_1/=len(lines)
    recall/=len(lines)
    open(w_path,"w").write(f"hit_1:{hit_1}\nrecall:{recall}")
def eval_hit_public_5(r_path,cnt):
    w_path=r_path.replace('infer_generate_main_','hit/').replace('_'+str(cnt)+'_test.json','.txt')
    lines=open(r_path).readlines()
    hit_1,hit_3=0,0
    recall=0.0
    for line in lines:
        sim_ranks=json.loads(line)["sim_ranks"]
        sft_index=json.loads(line)["sft_index"]
        loc=sim_ranks.index(sft_index+1)
        if sft_index+1 in sim_ranks[:1]:
            hit_1+=1
        if sft_index+1 in sim_ranks[:3]:
            hit_3+=1
        recall+=1/(loc+1)
    
    hit_1/=len(lines)
    hit_3/=len(lines)
    recall/=len(lines)
    open(w_path,"w").write(f"hit_1:{hit_1}\nhit_3:{hit_3}\nrecall:{recall}")
def eval_hit_public_6(r_path,cnt):
    w_path=r_path.replace('infer_generate_main_','hit/').replace('_'+str(cnt)+'_test.json','.txt')
    lines=open(r_path).readlines()
    hit_1,hit_3,hit_5=0,0,0
    recall=0.0
    for line in lines:
        sim_ranks=json.loads(line)["sim_ranks"]
        sft_index=json.loads(line)["sft_index"]
        loc=sim_ranks.index(sft_index+1)
        if sft_index+1 in sim_ranks[:1]:
            hit_1+=1
        if sft_index+1 in sim_ranks[:3]:
            hit_3+=1
        if sft_index+1 in sim_ranks[:5]:
            hit_5+=1
        recall+=1/(loc+1)
    
    hit_1/=len(lines)
    hit_3/=len(lines)
    hit_5/=len(lines)
    recall/=len(lines)
    open(w_path,"w").write(f"hit_1:{hit_1}\nhit_3:{hit_3}\nhit_5:{hit_5}\nrecall:{recall}")
import os
def eval_hit_dir_public(directory):
    for filename in os.listdir(directory):
        
        filepath = os.path.join(directory, filename)
        if filepath.endswith('.json'):
            cnt=filepath[-11]
            if int(cnt)<4:
               eval_hit_public_3(filepath,cnt)
            elif int(cnt)<=5:
               eval_hit_public_5(filepath,cnt)
            else:
               eval_hit_public_6(filepath,cnt)

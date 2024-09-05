import sys
sys.path.append("..")
import os
os.environ["TRANSFORMERS_CACHE"] = os.path.join("..","..","transformers_cache","models")
os.environ["HF_HOME"] = os.path.join("..","..","transformers_cache","datasets")
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
import nltk
TRANSFORMERS_OFFLINE=1
import torch
from llm2vec import LLM2Vec
import json
from tqdm import tqdm


    
def get_bleu(hyp, ref):
    hyp = hyp.strip()
    ref = ref.strip()
    return nltk.translate.bleu_score.sentence_bleu([ref], hyp)

def create_reward_fn_2():
    model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    model_device = "cuda:{}".format(torch.cuda.device_count() - 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "right"
    reward_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(model_device)
    reward_model.eval()

    def get_score(prefixes, suffixes):
        input_content = tokenizer(
            prefixes,
            suffixes,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(model_device)
        with torch.no_grad():
            rewards = reward_model(**input_content).logits
        
        return rewards.view(-1)
    
    return get_score, 140

def create_reward_fn_3():
    l2v = LLM2Vec.from_pretrained(
         "llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp",
         peft_model_name_or_path="llm2vec/LLM2Vec-Llama-2-7b-chat-hf-mntp-unsup-simcse",
        # device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        # device_map="cuda:2" ,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)

    def get_score(references, predictions):
        instruction = (
            "Given a question that may contain code, retrieve responses that are not only semantically relevant but also structurally relevant.")
        rewards=[]
        for reference,prediction in zip(references,predictions):
            reference= [[instruction,reference]]
            re_reps = l2v.encode(reference)
            pre_reps= l2v.encode([prediction])
            q_reps_norm = torch.nn.functional.normalize(re_reps, p=2, dim=1)
            a_reps_norm = torch.nn.functional.normalize(pre_reps, p=2, dim=1)
            cos_sim = torch.mm(q_reps_norm, a_reps_norm.transpose(0, 1)) 
            rewards.append(cos_sim)
        rewards=torch.stack(rewards).squeeze()
        return rewards.view(-1)
    
    return get_score

create_reward_fn = create_reward_fn_3
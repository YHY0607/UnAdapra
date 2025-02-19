from datasets import load_dataset
from datasets import Dataset
from utils.config import args
from dataclasses import dataclass
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    GPT2Tokenizer,
    DataCollatorWithPadding,
)
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
def normalization(new_vs):
    mean_vs=np.mean(new_vs)
    std_vs=np.std(new_vs)
    new_vs=(new_vs-mean_vs)/std_vs
    return new_vs
def min_max_norm(new_vs):
    new_vs=np.log(new_vs)
    max_v=max(new_vs)
    min_v=min(new_vs)
    new_vs=2*(new_vs-min_v)/(max_v-min_v)-1
    return new_vs
def norm_1(new_vs,new_min,new_max):
    min_val = np.min(new_vs, axis=0)
    max_val = np.max(new_vs, axis=0)
    normalized_data = (new_vs - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
    return normalized_data
def linear_decay_year(vs,ts):
    #qt is the time of query
    #vs is a list of votes
    #ts is a list of time
    datas=[]
    orders=[i+1 for i in range(len(vs))]
    new_vs=[]
    for v,t,order in zip(vs,ts,orders):
        new_time_obj=datetime.now()
        answer_time_obj= datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
        # difference_time= (new_time_obj-answer_time_obj).total_seconds()
        difference = relativedelta(new_time_obj, answer_time_obj)
        difference_months = difference.years * 12 + difference.months
        difference_years = difference.years
        # if int(v)==0:
        #     v=1e-4
       
        new_v=float(v)/(difference_years+1)
        new_vs.append(new_v)
        datas.append({order:new_v})
    sorted_dict_list = sorted(datas, key=lambda x: list(x.values())[0],reverse=True)
    sorted_keys = [list(d.keys())[0] for d in sorted_dict_list]
    new_orders=[]
    for i in orders:
      i_index=sorted_keys.index(i)+1
      new_orders.append(i_index)
    return new_orders,new_vs
    
    
def time_norm(vs,ts,qt):
    #qt is the time of query
    #vs is a list of votes
    #ts is a list of time
    datas=[]
    orders=[i+1 for i in range(len(vs))]
    new_vs=[]
    for v,t,order in zip(vs,ts,orders):
        query_time_obj = datetime.strptime(qt, "%Y-%m-%dT%H:%M:%S.%f")
        new_time_obj=datetime.now()
        answer_time_obj= datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%f")
        difference_time= (new_time_obj-answer_time_obj).total_seconds()
        if int(v)==0:
            v=1e-4
        new_v=float(v)/difference_time
        new_vs.append(new_v)
        datas.append({order:new_v})
    sorted_dict_list = sorted(datas, key=lambda x: list(x.values())[0],reverse=True)
    sorted_keys = [list(d.keys())[0] for d in sorted_dict_list]
    new_orders=[]
    for i in orders:
      i_index=sorted_keys.index(i)+1
      new_orders.append(i_index)
    return new_orders,min_max_norm(new_vs)
   
class DataManager():
    def __init__(self, config, training_stage, tokenizer_path = args.model_name_or_path):
        self.config = config
        if self.config.architectures[0].lower() == "llamaforcausallm":
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            self.tokenizer.unk_token = "<unk>"
            self.tokenizer.bos_token = "<s>"
            self.tokenizer.eos_token = "</s>"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.padding = True
        self.max_length = args.block_size
        self.pad_to_multiple_of = 8
        self.return_tensors = "pt"
        self.add_special_tokens = True
        self.training_stage = training_stage
        self.stop_sequences = ["\n\n"]
    
    def batch_decode(self, model_output):
        # model_output = [batch, seq_len]
        return self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

    def early_truncation(self, text):
        for stop in self.stop_sequences:
            stop_ix = text.find(stop)
            if stop_ix >= 0:
                text = text[:stop_ix].strip()
        return text.strip()
    def get_ranks(self,values1, values2):
        combined_values = list(zip(values1, values2))
        sorted_combined_values = sorted(combined_values, key=lambda x: (x[0], x[1]), reverse=True)
        ranking_dict = {value: idx + 1 for idx, value in enumerate(sorted_combined_values)}
        rankings = [ranking_dict[(v1, v2)] for v1, v2 in combined_values]
    
        return rankings
    def train_data_collator(self, features):
        samples_num = len(features)
        training_stage = self.training_stage
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)

        self.tokenizer.truncation_side = "left"
        ps = []
        ss = []
        rs = []
        se_ranks = [] # se_rank
        vs=[] # vote value
        vote_ranks=[]
        sft_index = []
        for feature_index, feature in enumerate(features):
            new_v=[]
            for v in feature['vote'][:training_stage]:
                new_v.append(int(v))

            se_rank=self.get_ranks(feature['reward'][:training_stage],new_v)
            vote_rank=self.get_ranks(new_v,feature['reward'][:training_stage])
            for p, s, se,v in zip(feature['prefix'][:training_stage], feature['suffix'][:training_stage], feature['reward'][:training_stage],feature['vote'][:training_stage]):
                p = "".join(p)
                p = p.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()
                ps.append(p)
                ss.append(s)
                rs.append(se)
                vs.append(int(v))
            se_ranks.append(se_rank)
            vote_ranks.append(vote_rank)
            assert feature["sft_index"] < training_stage
            sft_index.append(feature["sft_index"])

        ps = self.batch_decode(
            self.tokenizer(
                ps,
                max_length = self.max_length - 128,
                truncation = True,
                add_special_tokens = self.add_special_tokens,
            )['input_ids']
        )

        ps_input_ids = self.tokenizer(
            ps,
            add_special_tokens = self.add_special_tokens,
        )['input_ids']
        ps_lens = [len(p_input_ids)-1 for p_input_ids in ps_input_ids]
        
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"
        
        texts = []
        for p, s in zip(ps, ss):
            texts.append(p + " " + s)
        
        batch = self.tokenizer(
            texts,
            padding=self.padding,
            max_length = self.max_length,
            truncation = True,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        )
        
        seq_len = batch["attention_mask"].shape[1]
        prefix_mask = []
        for p_len in ps_lens:
            assert seq_len > p_len
            prefix_mask.append(
                [1 if i<p_len else 0 for i in range(seq_len)]
            )
        batch["prefix_mask"] = torch.tensor(prefix_mask)
        
        batch['labels'] = batch["input_ids"].clone().detach()
        for key in batch:
            batch[key] = batch[key].view(samples_num,training_stage,-1)
        
        batch['rewards'] = torch.tensor(rs).view(samples_num, -1)
        batch['se_rank'] = torch.tensor(np.array(se_ranks)).view(samples_num, -1) #[batch_size,train_stage]
        batch['vote_rank'] = torch.tensor(np.array(vote_ranks)).view(samples_num, -1) #[batch_size,train_stage]
        batch['vote'] = torch.tensor(np.array(vs)).view(samples_num, -1) #[batch_size,train_stage]
        batch['sft_index'] = torch.tensor(sft_index) # [batch]
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state

        return batch


    def load_train_data(
        self, 
        data_collator, 
        data_file_path, 
        data_file_name=None,
        extension='json', 
        stream = None, 
    ):
        raw_datasets = load_dataset(extension, data_dir = data_file_path, data_files = data_file_name, streaming=True if stream != None else False, split="train")

        dataloader = DataLoader(
            raw_datasets, 
            shuffle=True,
            collate_fn=data_collator, 
            batch_size=args.per_device_train_batch_size
        )

        return dataloader
    
    def infer_generate(self, model, prefixes,accelerator):
        model.to(accelerator.device)
        
        # prefixes = [prefix, prefix]
        origin_state = (self.tokenizer.padding_side, self.tokenizer.truncation_side)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"
        
        new_prefixes = []
        for p in prefixes:
            assert p[-7:] == "\n[END]:", p[-7:]
            p = p[:-7]
            new_prefixes.append(p)

        new_prefixes = self.batch_decode(
            self.tokenizer(
                new_prefixes,
                max_length = 512,
                truncation = True,
                add_special_tokens = self.add_special_tokens,
            )["input_ids"]
        )
        prefixes = [p + "\n[END]:" for p in new_prefixes]

        batch = self.tokenizer(
            prefixes,
            padding=self.padding,
            add_special_tokens = self.add_special_tokens,
            return_tensors = self.return_tensors,
        ).to(model.device)
        batch_size = len(prefixes)
        truncated_prefixes = self.batch_decode(batch['input_ids'])
        
        with torch.no_grad():
            predicted_sents = model.generate(
                **batch, 
                max_new_tokens = 64,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                num_return_sequences = 1,
            )
        
        instant_text = self.batch_decode(predicted_sents)
        
        # restore states
        self.tokenizer.padding_side, self.tokenizer.truncation_side = origin_state
        
        for index in range(len(instant_text)):
            assert truncated_prefixes[index].rstrip() in instant_text[index], (truncated_prefixes[index].strip(), instant_text[index])
            instant_text[index] = instant_text[index].replace(truncated_prefixes[index].rstrip(), "").strip()
            instant_text[index] = self.early_truncation(instant_text[index])
            
        return instant_text
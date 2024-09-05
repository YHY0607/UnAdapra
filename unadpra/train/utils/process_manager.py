
# import os
 
# # os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'
# # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# import torch
# from transformers import (
#     CONFIG_MAPPING,
#     MODEL_MAPPING,
#     AutoConfig,
#     AutoModelForCausalLM,
# )
# # # torch.cuda.set_device(4)
# # device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# # config= AutoConfig.from_pretrained('/home/yanghongyu/pro/llama-2-7b-chat-hf')
# # torch.cuda.empty_cache()
# # model= AutoModelForCausalLM.from_pretrained('/home/yanghongyu/pro/llama-2-7b-chat-hf',config=config).cuda(2)
# # print(model.device)
# # device1=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# # print(device)
# # model=model.to(device1)
# # print(model.device)
# # print(torch.cuda.device_count()) #可用GPU数量

import os
import json
from utils.config import args
import math
from scipy.stats import poisson
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn 
from accelerate.logging import get_logger
from .data_manager import  DataManager
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
)
from scipy.stats import spearmanr, kendalltau

from unadpra.train.utils.metrics import create_reward_fn

def min_max_norm(scores):
    min_score = torch.min(scores)
    max_score = torch.max(scores)
    scores = 2*(scores - min_score) / (max_score - min_score)-1
    return scores
class ProcessManager():
    def __init__(
        self,
        accelerator,
        model_path = args.model_name_or_path,
    ):
        self.accelerator = accelerator
        self.model_path = model_path
      
        self.model_config = AutoConfig.from_pretrained(self.model_path)
        
       
        self.data_manager = DataManager(
                self.model_config,
                args.training_stage_num,
            )
      
        
        # set model
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          config=self.model_config, 
                                                        #   device_map = 'auto',
                                                          torch_dtype = torch.float16
                                                          )
      

        self.model.resize_token_embeddings(len(self.data_manager.tokenizer))
        
        self.logger = get_logger(__name__)
    def compute_lambda_loss_se(self,scores,gains,ranks):
        batch_size = scores.shape[0]
        loss = 0.0
        deltas=[]
        losses=[]
        s=[]
        for b in range(batch_size):
            score = min_max_norm(scores[b])
            gain = gains[b]
            rank = ranks[b]
            delta_b=[]
         
            for i in range(len(score)):
                delta_i=[]
              
                for j in range(len(score)):
                    if i != j:
                        gain_i=torch.pow(2,gain[i])-1
                        gain_j=torch.pow(2,gain[j])-1
                        delta_ij = torch.abs(gain_i - gain_j) * torch.abs(1 / torch.log2(1 + rank[i]) - 1 / torch.log2(1 + rank[j]))
                        delta_i.append(delta_ij)
                delta_i = torch.stack(delta_i,dim=-1).unsqueeze(0) #[1,train_stage_num-1]
                delta_b.append(delta_i)
                
               
            delta_b = torch.cat(delta_b).unsqueeze(0) #[1,train_stage_num,train_stage_num-1]
            deltas.append(delta_b)
          
        # self.logger.info(f'Deltas: {deltas}')
   
        deltas = torch.cat(deltas, dim=0) #[batch_size,train_stage_num,train_stage_num-1]'
        # self.logger.info(f'se_Deltas: {deltas}')

        return deltas
    def compute_lambda_loss_vote(self,scores,gains,ranks):
        batch_size = scores.shape[0]
        loss = 0.0
        deltas=[]
        losses=[]
        s=[]
        for b in range(batch_size):
            score = min_max_norm(scores[b])
            gain = gains[b]
            rank = ranks[b]
            delta_b=[]
         
            for i in range(len(score)):
                delta_i=[]
              
                for j in range(len(score)):
                    if i != j:
                        # gain_i=torch.pow(2,gain[i])-1
                        # gain_j=torch.pow(2,gain[j])-1
                        delta_ij = torch.abs(gain[i] - gain[j]) * torch.abs(1 / torch.log(1 + rank[i]) - 1 / torch.log(1 + rank[j]))
                        delta_i.append(delta_ij)
                delta_i = torch.stack(delta_i,dim=-1).unsqueeze(0) #[1,train_stage_num-1]
                delta_b.append(delta_i)
                
               
            delta_b = torch.cat(delta_b).unsqueeze(0) #[1,train_stage_num,train_stage_num-1]
            deltas.append(delta_b)
        # self.logger.info(f"vote: {gains}")  
        # self.logger.info(f"vote_rank: {ranks}")
        # self.logger.info(f'vote_Deltas: {deltas}')
   
        deltas = torch.cat(deltas, dim=0) #[batch_size,train_stage_num,train_stage_num-1]
        return deltas
    def compute_lambda_loss_vote(self,scores,gains,ranks):
        batch_size = scores.shape[0]
        loss = 0.0
        deltas=[]
        losses=[]
        s=[]
        for b in range(batch_size):
            score = min_max_norm(scores[b])
            gain = gains[b]
            rank = ranks[b]
            delta_b=[]
         
            for i in range(len(score)):
                delta_i=[]
              
                for j in range(len(score)):
                    if i != j:
                        gain_i=torch.log10(gain[i]+1)
                        gain_j=torch.log10(gain[j]+1)
                        delta_ij = torch.abs(gain_i- gain_j) * torch.abs(1 / torch.log(1 + rank[i]) - 1 / torch.log(1 + rank[j]))
                        delta_i.append(delta_ij)
                delta_i = torch.stack(delta_i,dim=-1).unsqueeze(0) #[1,train_stage_num-1]
                delta_b.append(delta_i)
                
               
            delta_b = torch.cat(delta_b).unsqueeze(0) #[1,train_stage_num,train_stage_num-1]
            deltas.append(delta_b)
        # self.logger.info(f"vote: {gains}")  
        # self.logger.info(f"vote_rank: {ranks}")
        # self.logger.info(f'vote_Deltas: {deltas}')
   
        deltas = torch.cat(deltas, dim=0) #[batch_size,train_stage_num,train_stage_num-1]
        return deltas
    def get_ranks(self,values1, values2):
        combined_values = list(zip(values1, values2))
        sorted_combined_values = sorted(combined_values, key=lambda x: (x[0], x[1]), reverse=True)
        ranking_dict = {value: idx + 1 for idx, value in enumerate(sorted_combined_values)}
        rankings = [ranking_dict[(v1, v2)] for v1, v2 in combined_values]
    
        return rankings
    def batch_kendalltau(self,b_ranks,vote_ranks):
        batch_bias=[]
        for rank,vote_rank in zip(b_ranks,vote_ranks):
            tau, _ = kendalltau(rank.cpu().numpy(), vote_rank.cpu().numpy())
            batch_bias.append(torch.tensor(tau))
        batch_bias=torch.stack(batch_bias).to(self.accelerator.device) 
        return batch_bias
    # def dynamic_sft_rank(self,se_ranks,vote_ranks,se_weight,vote_weight):
    #     max_keys=[]
    #     train_stage_num = se_ranks.shape[1]
    #     for se_rank,vote_rank in zip(se_ranks,vote_ranks):
    #         scores = {i+1: 0 for i in range(train_stage_num)}
    #         for idx, rank in enumerate(se_rank):
    #            scores[idx+1] += se_weight*(train_stage_num - idx)
    #         for idx, rank in enumerate(vote_rank):
    #            scores[idx+1] += vote_weight*(train_stage_num+1 - rank)
    #         max_key = max(scores, key=scores.get)
    #         max_keys.append(torch.tensor(max_key))
    #     max_keys=torch.stack(max_keys)
    #     return max_keys
    def fill_diagonal(self,sv_delta,train_num_stage):
        fill_sv=[]
        for sv in sv_delta:
            weight = torch.zeros((train_num_stage, train_num_stage), device=self.accelerator.device)

            for i in range(train_num_stage):
                if 1:
                    weight[i, :i] = sv[i, :i]
                    weight[i, i+1:] = sv[i, i:]
            fill_sv.append(weight)
        return torch.cat(fill_sv).unsqueeze(0)
    def get_dy_rankings(self,sv_delta,se_ranks,train_num_stage):
        sfts=[]
        for weight,se_rank in zip(sv_delta,se_ranks):
            
            sft=[]
            for i in range(train_num_stage-1):
                lambda_max=torch.max(weight)
                index=torch.where(weight==lambda_max)
                row=index[0][0]
                col=index[1][0]
                if se_rank[row]<se_rank[col]:
                    sft.append(row.item())
                    weight[:,row]=0
                    weight[row,:]=0
                else:
                    sft.append(col.item())
                    weight[:,col]=0
                    weight[col,:]=0
            for i in range(train_num_stage):
                if i not in sft:
                    sft.append(i)
            sft=torch.tensor(sft).unsqueeze(0).to(self.accelerator.device)
            sfts.append(sft)
        sfts=torch.cat(sfts)
        return sfts #[bs,train_num_stage]        


    def lambda_sft_index(self,sv_delta,se_rank):
        se_rank=se_rank.squeeze(0)
        lambda_max=torch.max(sv_delta)
        index=torch.where(sv_delta==lambda_max)[0]
        if se_rank[index[0]]<se_rank[index[1]]:
            return torch.tensor([index[0]]).unsqueeze(0)
        else:
            return torch.tensor([index[1]]).unsqueeze(0)

    def compute_loss(self, model, batch, print_loss,step):
        """
            batch = [batch, training_stage, seq_len]
        """        
        batch.to(self.accelerator.device)
        model.to(self.accelerator.device)
        batch_size = batch["labels"].shape[0]
        temp_training_stage = batch["labels"].shape[1]
        sub_batches = [{key: batch[key][:,time,:] for key in ["input_ids", "attention_mask"]} for time in range(temp_training_stage)]
        
        score_list = []
        suffix_mask_list = []
        # print(model.device)
        # print(sub_batch.device)
        for batch_index, sub_batch in enumerate(sub_batches):
            local_outputs = model(**sub_batch, output_hidden_states=True, return_dict=True)
            local_logits = local_outputs.logits #[batch, seq_len, token_num] [2,520,32000]
            local_mask = sub_batch["attention_mask"] & (~batch["prefix_mask"][:, batch_index, :]) #[batch, seq_len]
            local_labels = batch["labels"][:, batch_index, :]

            # Shift
            shift_logits = local_logits[..., :-1, :].contiguous() #[batch, seq_len-1, token_num]
            shift_logits = F.log_softmax(shift_logits, dim=2) #[batch, seq_len-1, token_num]
            shift_masks = local_mask[..., :-1] #[batch, seq_len-1]
            shift_labels = local_labels[..., 1:].view(batch_size, -1, 1) #[batch, seq_len-1, 1]

            selected_logits = torch.gather(input=shift_logits, dim=2, index=shift_labels).view(batch_size, -1) #[batch, seq_len-1]
            selected_logits[shift_masks != 1] = 0.0 #[batch, seq_len-1]
            sentence_logits = torch.sum(selected_logits, dim=1) #[batch]
            sentence_logits = sentence_logits.view(batch_size, 1)
            score_list.append(sentence_logits)
            suffix_mask_list.append(torch.sum(shift_masks, dim=1).view(batch_size, 1))
        
        sum_scores = torch.cat(score_list, dim=1) #[batch, training_stage], check the sum  score rank is not equals to the simi_orders
        suffix_mask = torch.cat(suffix_mask_list, dim=1) #[batch, training_stage]
        scores = sum_scores / suffix_mask #[batch, training_stage]
        se_ranks=batch["se_rank"]
        vote_ranks=batch["vote_rank"]
        total_loss = 0.0
        # use the static ranks
        votes=batch["vote"]
        # self.logger.info(f"vote: {votes}")
        se_delta= self.compute_lambda_loss_se(scores,batch["rewards"],se_ranks) #[batch_size,train_stage_num,train_stage_num-1]
        vote_delta=self.compute_lambda_loss_vote(scores,votes,vote_ranks)
        sv_delta=se_delta*vote_delta
        se_vote_losses = []
        
        sv_delta=self.fill_diagonal(sv_delta,temp_training_stage)
        se_delta=self.fill_diagonal(se_delta,temp_training_stage)
        vote_delta=self.fill_diagonal(vote_delta,temp_training_stage)
        new_sv_delta=sv_delta.clone()
        ranks=self.get_dy_rankings(sv_delta,se_ranks,temp_training_stage) #[bs,train_stage_num-1]
        for time in range(temp_training_stage-1):
            # neg_reward = batch["rewards"][:, time+1:] # [batch, training_stage-time-1]
            # pos_reward = batch["rewards"][:, time] # [batch]
            best_index=ranks[:,time].item()
            worst_index=ranks[:,-1].item()
            # semantic*vote loss
            eps = 1e-10
            neg_temperatures_se = se_delta[:,best_index,:] #[bs,train_stage_num]
            pos_temperature_se = torch.max(neg_temperatures_se, dim=1).values # [batch]
            # pos_temperature_se =  neg_temperatures_se[:,worst_index].item() # [batch]
           
            
            neg_temperatures_vote = vote_delta[:,best_index,:] #[bs,1,train_stage_num-1]
            pos_temperature_vote = torch.max(neg_temperatures_vote, dim=1).values # [batch]
            other_fenmu=torch.sum(torch.exp(scores * neg_temperatures_vote*neg_temperatures_se), dim=1)

            other_fenmu1=torch.sum(torch.exp(scores[:, :best_index] * neg_temperatures_vote[:,:best_index]*neg_temperatures_se[:,:best_index]), dim=1)
            other_fenmu2=torch.sum(torch.exp(scores[:, best_index+1:] * neg_temperatures_vote[:,best_index+1:]*neg_temperatures_se[:,best_index+1:]), dim=1)
            loss = torch.log(eps + torch.exp(scores[:, best_index] * pos_temperature_vote*pos_temperature_se) + other_fenmu1+other_fenmu2) - scores[:, best_index] * pos_temperature_se*pos_temperature_vote # [batch]
            se_vote_loss = torch.mean(loss).to(local_outputs.hidden_states[0].dtype)
            se_vote_losses.append(se_vote_loss.item())
            total_loss += se_vote_loss
        dy_sft_index=ranks[:,0].unsqueeze(0)
        dy_sft_scores = torch.gather(input = sum_scores, dim = 1, index = dy_sft_index).view(batch_size) 
        dy_sft_loss = torch.mean(-dy_sft_scores).to(local_outputs.hidden_states[0].dtype)
        dy_sft_loss = args.sft_weight  * dy_sft_loss
        total_loss+=dy_sft_loss
        if step % args.gradient_accumulation_steps==0:
            self.logger.info(f"Step {step}")
            self.logger.info(f"se_delta: {se_delta}")
            self.logger.info(f"vote_delta: {vote_delta}")
            self.logger.info(f"sv_delta: {new_sv_delta}")
            self.logger.info(f"Semantic_vote Loss: {se_vote_losses}")
            self.logger.info(f"se_ranks: {se_ranks}")
            self.logger.info(f"vote_ranks: {vote_ranks}")
            self.logger.info(f"ranks: {ranks}")
            self.logger.info(f"dy_sft_index: {dy_sft_index}")
            self.logger.info(f"dy_sft_loss: {dy_sft_loss}")
            self.logger.info(f"Total loss: {total_loss}")
      
        return total_loss

    def prepare_hfa_dataloader(self, train_file_path=args.train_file_path, train_file_name = None):
        # get dataloader
        if train_file_name == None:
            train_file_name = os.listdir(train_file_path)[0]
        
        self.logger.info(f"Load training data from {os.path.join(train_file_path, train_file_name)}")
        self.accelerator.print(f"Load training data from {os.path.join(train_file_path, train_file_name)}")
        
        hfa_dataloader = self.data_manager.load_train_data(
            data_file_path = train_file_path,
            data_file_name = train_file_name,
            data_collator = self.data_manager.train_data_collator
        )
        
        # wrap with accelerator
        hfa_dataloader = self.accelerator.prepare(
            hfa_dataloader
        )

        return hfa_dataloader

    def init_prepare_train(self, train_file_name = None):
        # get dataloader
        train_files = os.listdir(args.train_file_path)
        
        if train_file_name == None:
            train_file_name = train_files[0]
        
        # record raw dataset length
        dataset_length = len(
            open(os.path.join(args.train_file_path, train_file_name), 'r', encoding='utf-8').readlines()
        )

        # get the placeholder dataloader
        placeholder_dataloader = self.data_manager.load_train_data(
            data_file_path = args.train_file_path,
            data_file_name = train_file_name,
            data_collator = self.data_manager.train_data_collator
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        
        # Scheduler and math around the number of training steps.
        if args.max_train_steps is None:
            num_update_steps_per_epoch_per_train_file = math.ceil(
                math.ceil(
                    dataset_length / args.per_device_train_batch_size
                ) / args.gradient_accumulation_steps
            )
            args.max_train_steps = len(train_files) * num_update_steps_per_epoch_per_train_file * args.num_train_epochs
        

        model, optimizer, _ = self.accelerator.prepare(
            self.model, optimizer, placeholder_dataloader
        )
        
        # self.model = None

        total_batch_size = args.per_device_train_batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        self.logger.info("***** Running training *****", main_process_only=True)
        self.logger.info(f"  Num examples = {len(train_files) * dataset_length}", main_process_only=True)
        self.logger.info(f"  Num training stages = {args.training_stage_num}", main_process_only=True)
        self.logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}", main_process_only=True)
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}", main_process_only=True)
        self.logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}", main_process_only=True)
        
        return model, optimizer, dataset_length

    def train(self):
        train_files = os.listdir(args.train_file_path)
        model, optimizer,  dataset_length = self.init_prepare_train(
            train_file_name = train_files[0]
        )
        # model.cuda(2)
        training_stage = args.training_stage_num
        if self.accelerator.is_main_process:
            if args.do_validation:
                get_score,_ = create_reward_fn()
            writer = SummaryWriter(args.log_path)
        
         
        self.accelerator.wait_for_everyone()
        # Train!
        progress_bar = tqdm(
            range(
                len(train_files) * math.ceil(
                    math.ceil(
                        math.ceil(
                            dataset_length / args.per_device_train_batch_size
                        ) / self.accelerator.num_processes
                    ) / args.gradient_accumulation_steps
                ) * args.num_train_epochs
            ),
            disable=not self.accelerator.is_local_main_process
        )
        last_dev_reward=0.0
        for epoch in range(args.num_train_epochs):
            total_loss = 0.0
            if self.accelerator.is_main_process:
                self.logger.info(f"Epoch {epoch} starts")
                self.accelerator.print(f"\nEpoch {epoch} starts")
            train_file_path = args.train_file_path

            for train_file_index, train_file_name in enumerate(train_files):
                if_get_new_dataloader = False
                if len(train_files) == 1:
                    if epoch == 0:
                        if_get_new_dataloader = True
                    else:
                        pass
                else:
                    if_get_new_dataloader = True
                
                torch.cuda.empty_cache()
                if if_get_new_dataloader:
                    hfa_dataloader = None
                    hfa_dataloader = self.prepare_hfa_dataloader(train_file_path, train_file_name)
                
                print_loss = []
                with self.accelerator.accumulate(model):
                    model.to(self.accelerator.device)
                    model.train()
                    for step, batch in enumerate(hfa_dataloader):
                        if step!=0 and step % args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            self.logger.info(f'Total local training loss is: {total_loss}', main_process_only=True)
                            total_loss=0.0

                        if step !=0 and step % args.checkpointing_step == 0:
                            model_to_save = self.accelerator.unwrap_model(model)
                            dev_res = self.infer(
                                model = model_to_save,
                                infer_file_path = args.validation_file_path,
                                infer_file_name = args.validation_file_name,
                                   )
                            references, predictions = [], []
                            batch_size = args.per_device_eval_batch_size
                            dev_reward = 0
                            for index, sample in enumerate(dev_res):
                                reference= sample['suffix'][0]
                                prediction = sample['infer']["t"].strip()
                                references.append(reference)
                                predictions.append(prediction)
                                if len(references) == batch_size or index == len(dev_res)-1:
                                    batch_rewards=get_score(references,predictions)
                                    batch_rewards = torch.sigmoid(batch_rewards).cpu().detach().numpy().tolist() #[batch_size]
                                    dev_reward += sum(batch_rewards)
                                references, predictions = [], []
                               
                            dev_reward = dev_reward / len(dev_res)
                            self.logger.info(f"Step {step} | Dev avg reward {dev_reward}")
                            if dev_reward > last_dev_reward:
                                last_dev_reward=dev_reward

                                self.logger.info(f"best Step {step} checkpoint with higher Dev avg reward (the best checkpoint so far)")
                                last_dev_reward = dev_reward
                                self.save_checkpoint(model_to_save,self.data_manager.tokenizer,os.path.join(args.output_dir, 'epoch_{}_step_{}'.format(epoch,step)))
                                self.logger.info(f"Step {os.path.join(args.output_dir, 'epoch_{}_step_{}'.format(epoch,step))} checkpoint has been saved.")
                     
                       
                      
                      
                        loss=self.compute_loss(model, batch, print_loss,step)
                     
                      
                        self.accelerator.backward(loss)

                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        
                        optimizer.zero_grad()
                        if self.accelerator.is_local_main_process:
                            total_loss += loss.item() 
                        # self.logger.info((f"optimizer device: {optimizer.device}"))
                # self.logger.info(f'Total local training loss is: {total_loss}', main_process_only=True)
                self.logger.info('Do eval.......')
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_local_main_process:
                    model_to_save = self.accelerator.unwrap_model(model)
                    model.eval()
                    dev_res = self.infer(
                        model = model_to_save,
                        infer_file_path = args.validation_file_path,
                        infer_file_name = args.validation_file_name,)
                    references, predictions = [], []
                    batch_size = args.per_device_eval_batch_size
                    dev_reward = 0
                    for index, sample in enumerate(dev_res):
                        reference= sample['suffix'][0]
                        prediction = sample['infer']["t"].strip()
                        references.append(reference)
                        predictions.append(prediction)
                        # if len(references) == batch_size or index == len(dev_res)-1:
                        batch_rewards = torch.sigmoid(get_score(references,predictions)).cpu().detach().numpy().tolist() #[batch_size]
                        dev_reward += sum(batch_rewards)
                        references, predictions = [], []
                    dev_reward = dev_reward / len(dev_res)
                    self.logger.info(f"Epoch {epoch} | Dev avg reward {dev_reward}")
                  
                    if dev_reward > last_dev_reward:
                        last_dev_reward=dev_reward
                        self.logger.info(f"epoch {epoch} checkpoint with higher Dev avg reward (the best checkpoint so far)")
                        self.save_checkpoint(model_to_save,self.data_manager.tokenizer,os.path.join(args.output_dir, 'epoch_{}'.format(epoch)))
                        self.logger.info(f"epoch {os.path.join(args.output_dir, 'epoch_{}'.format(epoch))} checkpoint has been saved.")
                  

                        self.accelerator.wait_for_everyone()
                torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            writer.close()
        
        return self.accelerator.unwrap_model(model)
    
    def infer(self, model, infer_file_path=None, infer_file_name=None):
        torch.cuda.empty_cache()
        model.eval()

        with open(os.path.join(infer_file_path, infer_file_name), "r", encoding='utf-8') as f:
            infer_data = [json.loads(l) for l in f.readlines()]

        # sort
        length = []
        for l in infer_data:
            lens = 0
            for p in l['prefix'][0]:
                lens += (len(p.split(" ")))
            length.append(lens)
        
        indices = list(range(len(length)))
        back_indices = indices
        infer_data = [infer_data[index] for index in indices]
        
        infer_batch_size = args.per_device_eval_batch_size                                
        infer_bar = tqdm(range(len(infer_data)), desc= "Inference on {}".format(infer_file_name))
        for sample_index in range(0,len(infer_data),infer_batch_size):
            if len(infer_data)-sample_index < infer_batch_size:
                infer_batch_size = len(infer_data)-sample_index

            prefixes = [l['prefix'][0] for l in infer_data[sample_index:sample_index+infer_batch_size]]
            suffixes = self.data_manager.infer_generate(model, prefixes,self.accelerator)
            for l, s in zip(infer_data[sample_index:sample_index+infer_batch_size], suffixes):
                l['infer'] = {"t": s}
            infer_bar.update(infer_batch_size)
        torch.cuda.empty_cache()
        infer_data = [infer_data[index] for index in back_indices]

        return infer_data

    def save_checkpoint(self, model, tokenizer, path):
        if path is not None and path != '':
            os.makedirs(path, exist_ok=True)
            tokenizer.save_pretrained(path)
            model.save_pretrained(
                path, 
                is_main_process=self.accelerator.is_main_process, 
                save_function=self.accelerator.save,
            )
        else:
            self.logger.error('No save path!', main_process_only=True)
            
    # def save_checkpoint(self, tokenizer, path):
    #     if path is not None and path != '':
    #         os.makedirs(path, exist_ok=True)
    #         tokenizer.save_pretrained(path)
    #         model= self.accelerator.unwrap_model(self.model)
    #         # optimizer=self.accelerator.unwrap_optimizer(self.optimizer)
    #         # lr=unwrap_lr =self. accelerator.unwrap_model(scheduler)
    #         model.save_pretrained(
    #             path, 
    #             is_main_process=self.accelerator.is_main_process, 
    #             save_function=self.accelerator.save,
    #         )
    #         optimizer.save_pretrained(
    #             path, 
    #             is_main_process=self.accelerator.is_main_process, 
    #             save_function=self.accelerator.save,
    #         )
    #     else:
    #         self.logger.error('No save path!', main_process_only=True)
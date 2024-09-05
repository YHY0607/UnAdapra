#import some packages and reward funcs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import json
import tqdm
import torch
import torch.nn.functional as F
import unadpra.eval.metrics1 as metrics1
import metrics2 as metrics2

from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM
)
from peft import PeftConfig, PeftModel
from infer_func_now import setup_seed
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta


def eval(file_name):
    setup_seed()
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # rank = int(os.environ['RANK'])
    rank_sum = accelerator.num_processes
    torch.cuda.empty_cache()
    save_path = file_name.replace('reward','eval')

    # print(f"Rank {rank} is activated...")
    if accelerator.is_main_process:
      
        
        with open(file_name, 'r', encoding='utf-8') as f:
            infer_data = [json.loads(l) for l in f.readlines()]
        # if "line_index" in infer_data[0]:
        infer_data = {index: l for index,l in enumerate(infer_data)}
        # with open(save_path, 'w', encoding='utf-8') as f:
        #     infer_data = [infer_data[line_index] for line_index in range(len(infer_data))]
        #     for line in infer_data:
        #         content = json.dumps(line, ensure_ascii=False)
        #         f.write(content+'\n')

    accelerator.wait_for_everyone()

    get_code = metrics1.create_reward_fn()
    get_reward,_=metrics2.create_reward_fn()
    with open(file_name, 'r', encoding='utf-8') as f:
        infer_data = [json.loads(l) for line_index, l in enumerate(f.readlines())]
    raw_prefixes = [l['prefix'][0].strip() + " " for l in infer_data]
    # reference_suffixes = [ l['suffix'][int(l['gpt_ranks'][0]-1)] for l in infer_data]
    if 'llama3' in file_name:
        generated_suffixes = [l['infer']["t"]['content'].strip() for l in infer_data]
    else:
        generated_suffixes = [l['infer']["t"].strip() for l in infer_data]
    rank=torch.device("cuda")
    setup_seed()
    rewards = []
    codes=[]
    batch_size = 1
    for index in tqdm.tqdm(range(0,len(raw_prefixes), batch_size), desc=f"Rank {rank} rewarding..."):
        if len(raw_prefixes) - index < batch_size:
            batch_size = len(raw_prefixes) - index
        codes.extend(torch.sigmoid(get_code(raw_prefixes[index:index+batch_size], generated_suffixes[index:index+batch_size])).cpu().detach().numpy().tolist())
        rewards.extend(torch.sigmoid(get_reward(raw_prefixes[index:index+batch_size], generated_suffixes[index:index+batch_size])).cpu().detach().numpy().tolist())
    assert len(rewards) == len(generated_suffixes) and len(rewards) == len(infer_data), (len(rewards), len(generated_suffixes), len(infer_data))

    for index in range(len(infer_data)):
        reference_index = int(infer_data[index]['gpt_ranks'][0]-1)
        infer_data[index]["infer"]["q_code_sim"] = codes[index]
        infer_data[index]["infer"]["reward"] = rewards[index]
        if 'llama3' in file_name:
            infer_data[index]["infer"]["bleu"] = metrics2.get_bleu(infer_data[index]['infer']['t']['content'], infer_data[index]['suffix'][reference_index])
        else:
            infer_data[index]["infer"]["bleu"] = metrics2.get_bleu(infer_data[index]['infer']['t'], infer_data[index]['suffix'][reference_index])
    
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in infer_data:
            content = json.dumps(line, ensure_ascii=False)
            f.write(content+'\n')
    print(f"Rank {rank} completed!")
if __name__ == "__main__":
    directory = "./reward"
    for filename in os.listdir(directory):
        
        filepath = os.path.join(directory, filename)
        if filepath.endswith('.json'):
            eval(filepath)
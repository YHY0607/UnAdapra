#import some packages and reward funcs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import json
import tqdm
import torch
import torch.nn.functional as F
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

def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--index', type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--directory', default="", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    setup_seed()
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    # rank = int(os.environ['RANK'])
    rank_sum = accelerator.num_processes
    torch.cuda.empty_cache()
    # print(f"Rank {rank} is activated...")
    if accelerator.is_main_process:
        file_name = "test.json"
        save_path = os.path.join("inference_res", "infer_main_{}_{}_{}".format(args.index, args.stage, file_name))
        if os.path.exists(save_path):
            os.remove(save_path)
            # os.makedirs(save_path)
        
        save_path = os.path.join("./inference_res/cache", "infer_generate_main_{}_{}_{}".format(args.index, args.stage, file_name))
        
        with open(save_path, 'r', encoding='utf-8') as f:
            infer_data = [json.loads(l) for l in f.readlines()]
        # if "line_index" in infer_data[0]:
        infer_data = {index: l for index,l in enumerate(infer_data)}
        with open(save_path, 'w', encoding='utf-8') as f:
            infer_data = [infer_data[line_index] for line_index in range(len(infer_data))]
            for line in infer_data:
                content = json.dumps(line, ensure_ascii=False)
                f.write(content+'\n')

    accelerator.wait_for_everyone()

    get_score, reward_batch_size = metrics2.create_reward_fn()

    file_name = "test.json"
    save_path = os.path.join("./inference_res/cache", "infer_generate_main_{}_{}_{}".format(args.index, args.stage, file_name))
    with open(save_path, 'r', encoding='utf-8') as f:
        infer_data = [json.loads(l) for line_index, l in enumerate(f.readlines())]
    raw_prefixes = [l['prefix'][0].strip() + " " for l in infer_data]
    generated_suffixes = [l['infer']["t"].strip() for l in infer_data]
    rank=torch.device("cuda")
    setup_seed()
    rewards = []
    batch_size = reward_batch_size
    for index in tqdm.tqdm(range(0,len(raw_prefixes), batch_size), desc=f"Rank {rank} rewarding..."):
        if len(raw_prefixes) - index < batch_size:
            batch_size = len(raw_prefixes) - index
        rewards.extend(torch.sigmoid(get_score(raw_prefixes[index:index+batch_size], generated_suffixes[index:index+batch_size])).cpu().detach().numpy().tolist())
    assert len(rewards) == len(generated_suffixes) and len(rewards) == len(infer_data), (len(rewards), len(generated_suffixes), len(infer_data))

    for index in range(len(infer_data)):
        infer_data[index]["infer"]["score"] = rewards[index]
        infer_data[index]["infer"]["bleu"] = metrics2.get_bleu(infer_data[index]['infer']['t'], infer_data[index]['suffix'][0])
    
    save_path = os.path.join("./inference_res", "infer_main_{}_{}_{}".format(args.index, args.stage, file_name))
    with open(save_path, 'a', encoding='utf-8') as f:
        for line in infer_data:
            content = json.dumps(line, ensure_ascii=False)
            f.write(content+'\n')
    print(f"Rank {rank} completed!")
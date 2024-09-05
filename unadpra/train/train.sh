export OMP_NUM_THREADS=16
root_dir=..

#stage 23
id=$1
ranking_len=$2
mkdir -p $root_dir/logs/$id/$ranking_len
accelerate launch --num_processes 1  --config_file config.yaml main.py \
    --train_file_path ../data/train \
    --validation_file_path ../data/dev \
    --validation_file_name ../valid.json \
    --output_dir ../output \
    --log_path $root_dir/logs/$id/$ranking_len \
    --index $id \
    --seed 42 \
    --temperature 1 \
    --sft_weight 0.05 \
    --num_train_epochs 4 \
    --training_stage_num $ranking_len \
    --block_size 512 \
    --learning_rate 5e-7 \
    --checkpointing_step 200 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --do_train \
    --do_validation > $root_dir/logs/$id/$ranking_len/train_detail.log 2>&1
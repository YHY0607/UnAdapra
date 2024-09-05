
id=$1
ranking_len=$2
accelerate launch --config_file config.yaml  generate.py \
    --index $id \
    --stage $ranking_len > logs/generate_infer_main_${id}_${ranking_len}.log 2>&1
accelerate launch --config_file config.yaml   reward_new.py 
python -u score.py 







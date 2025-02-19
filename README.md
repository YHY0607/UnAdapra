# Data Preprocessing and Training Pipeline

This guide provides a step-by-step process for preparing data and generating training data for your model. Follow the instructions below to preprocess your data and generate semantic ranks and final training data.

## 1. Data Preprocessing

The data preprocessing includes several important steps to filter and clean your dataset:

### Operations Performed:
1. **HTML to JSON:** Convert HTML data into JSON format for easier processing.
2. **Select Data with Accepted Answers:** Choose only the data entries that have accepted answers.
3. **Select Data with Code Blocks:** Filter data entries that contain code blocks.
4. **Choose Specific Ranking Length:** Select data entries with a specific ranking length.
5. **Choose High-Quality Data:** Filter and choose high-quality data by running the `choose_good_train_data` function.
6. **Clean Data:** Clean the data to remove any irrelevant or malformed entries.

### Running the Preprocessing Script

To perform the above operations, navigate to the `data` directory and run the preprocessing script:

```bash
cd data
python data_preprocess.py
```

## Training Data Generation

Once your data is preprocessed, you can proceed with generating the semantic ranks and final training data.

### Step 1: Generate Semantic Ranks and Training Data

After preprocessing the data, the next step is to generate the semantic ranks and the training data that will be used for model training. This process involves the following:

- **Semantic Rank Generation:** The script processes the cleaned data and assigns semantic ranks based on the chosen criteria.
- **Final Training Data Creation:** The generated semantic ranks are then used to create the final training data that will be input to the model.

### Running the Training Data Generation Script

To initiate this process, navigate to the directory where the script is located and execute the following command:

```bash
python generate_train_data.py

```
## 2. Training

To start training, execute the following command:

```bash
cd train
. train.sh $id $rank_len train.sh

```


## 3. Evaluation
To run evaluation, execute the following commands:

1. Run the inference generation process:
```bash
accelerate launch --config_file config.yaml generate.py \
    --index $id \
    --stage $ranking_len > logs/generate_infer_main_${id}_${ranking_len}.log 2>&1

```

2. Run the reward calculation process:
```bash
accelerate launch --config_file config.yaml reward_new.py

```

3. Finally, evaluate the score:
```bash
python -u score.py

```






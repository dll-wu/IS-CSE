# IS-CSE
This code can be used to reproduce the results of our paper ***Instance Smoothed Contrastive Learning for Unsupervised Sentence Embedding***(Our [paper](https://arxiv.org/abs/2305.07424) has been accepted to AAAI2023.)

## Dependencies
We run our code on NVIDIA A100 with CUDA version over 11.0.

You may use the command below
```shell
conda create -n iscse python=3.8
conda activate iscse

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## Dataset
We use the same training data and evaluation data as in [SimCSE](https://arxiv.org/abs/2104.08821). To download the data, please use the following instructions:

Training data:
```
cd data
bash download_wiki.sh
```

Evaluation data(STS tasks):
```
cd SentEval/data/downstream
bash download_dataset.sh
```


## Training
Our code for IS-CSE follows the [released code of SimCSE](https://github.com/princeton-nlp/SimCSE), from which we can get training data and evaluation data.


We offer the training scripts for 4 backbones: BERT-base, BERT-large, RoBERTa-base and RoBERTa-large.

For example, training IS-CSE-BERT-base:
```
bash run_unsup_bert_base.sh
```

## Evaluation
Modify the model path in run_eval.sh

To evaluate a different task, simply modify the ```--task_set```. ("sts" means STS tasks, "transfer" means transfer tasks and "full" means both STS tasks and transfer tasks.)

```bash
python evaluation.py \
    --model_name_or_path Path_to_model \
    --pooler cls_before_pooler \
	--task_set sts \
    --mode test
```

Then run the script:

```
bash run_eval.sh
```

## Model List
|              Model              | Avg. STS |
|:-------------------------------|:--------:|
|  [iscse-bert-base-constant-alpha-0.1](https://westlakeu-my.sharepoint.com/:u:/g/personal/hehongliang_westlake_edu_cn/ET5Kr0-8Vg5Hqfc29jMbjPoBqOwKmreIrsNxTRsQ8d2-8A?e=PMgpsR) |   78.30 |
| [iscse-bert-large-cos-alpha-0.005-0.05](https://westlakeu-my.sharepoint.com/:u:/g/personal/hehongliang_westlake_edu_cn/EaDCx7POFwZPldN7YeeSPucBs3VVIPytIHrX-cRzUA75Qw?e=sbm6cX) |   79.47  |
|    [iscse-roberta-base-constant-alpha-0.1](https://westlakeu-my.sharepoint.com/:u:/g/personal/hehongliang_westlake_edu_cn/EbfyJszBQmlPo_dck5qI3vQBMGVvVcZBS-s0FBtq69IJ8w?e=0rmcev)    |   77.73  |
|    [iscse-roberta-large-cos-alpha-0.005-0.05](https://westlakeu-my.sharepoint.com/:u:/g/personal/hehongliang_westlake_edu_cn/EdUgyrqbhwNJt7DrOba5ATABZwEOIJPVq2CvMAHFHfIQRQ?e=oPuIsT)   |   79.42  |

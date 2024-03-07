# Timm Model Finetuning

Contributor: Pinyuan Feng (Tony)

## Descriptiion

This repo is used to finetune the [Timm](https://timm.fast.ai/) models on ImageNet. [Accelerate](https://huggingface.co/docs/accelerate/en/index) library is adopted to achieve efficient, easy-to-deployed parallel computing. 

## Environment Setup

```
conda create -n hmn python=3.9 -y
conda activate timm
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install timm==0.9.0 
pip install wandb accelerate pathlib numpy tqdm scipy torchmetrics pandas matplotlib
```

## Commands
### Initialization
```
accelerate config
```
After that you will see a list of questions. Just set up the configuration based on your situation

### Dummy Training and Evaluation
Before starting your large-scale training, you can use following code to test your scrip:
```
accelerate launch --main_process_port 29501 main_accelerate_timm.py -dm True -mn 'resnet50.tv_in1k' -ep 10 -bs 16 -pt True

accelerate launch main_accelerate_timm.py dm True -mn 'resnet50.tv_in1k' -bs 64 -ev True -rs True -pt True
```

### Training and Evaluation
```
accelerate launch --main_process_port 29501 main_accelerate_timm.py -mn 'resnet50.tv_in1k' -ep 20 -bs 256 -pt True -wb True

accelerate launch main_accelerate_timm.py dm True -mn 'resnet50.tv_in1k' -bs 256 -ev True -rs True -bt "best_acc"
```

## Notice

- Enter your token and project name to enable W&B for logging
- Modify your data path and checkpoint path
- More data augmentation techniques could be added to enhance your training
- ```--main_process_port 29501``` could be removed if your first GPU (0) is not used by other programs.
- You can visit [this github page](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv) to find the models you want.
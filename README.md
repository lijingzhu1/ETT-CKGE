# ETT-CKGE

<p align="center">
  <img src="assets/new_overview.png" alt="ETT-CKGE Model Overview" width="600"/>
</p>

This repository contains the codes and datasets for our paper:  
**"ETT-CKGE: Efficient Task-driven Tokens for Continual Knowledge Graph Embedding"**,  
which has been **accepted at ECML PKDD 2025** 🎉.



## Folder Structure

The structure of the folder is shown below:

```csharp
 ETT-CKGE
 ├─checkpoint
 ├─data
 ├─logs
 ├─save
 ├─src
 ├─main.py
 ├─data_preprocess.py
 └README.md
```

Introduction to the structure of the folder:

- /checkpoint: The generated models are stored in this folder.
- /data: The datasets(ENTITY, RELATION, FACT, HYBRID, graph_equal, graph_higher, graph_lower) are stored in this folder.
- /logs: Logs for the training are stored in this folder.
- /save: Some temp results are in this folder.
- /src: Source codes are in this folder.
- /main.py: To run the IncDE.
- data_preprocess.py: To prepare the data processing.
- README.md: Instruct on how to realize FEMME.

## Requirements

All experiments are implemented on the GPU with the PyTorch. The version of Python is 3.7.

Please run as follows to install all the dependencies:

```shell
pip3 install -r requirements.txt
```

## Usage

### Preparation

1. Unzip the dataset $data1.zip$ and $data2.zip$ in the folder of $data$.
2. Prepare the data processing in the shell:


### Main Results

3. Run the code with this in the shell:
#### ENTITY
```shell

python main.py -dataset ENTITY -lifelong_name double_tokened -using_token_distillation_loss True -use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -learning_rate 0.001 -patience 3 -token_distillation_weight 5000 10000 10000 10000 -token_num 2 -div_loss_weight 0.2
```
#### FACT
```shell

python main.py -dataset FACT -lifelong_name double_tokened -using_token_distillation_loss True -use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -learning_rate 0.001 -patience 3 -token_distillation_weight 1000 10000 10000 10000 -token_num 4 -div_loss_weight 0.6
```
#### HYBRID
```shell

python main.py -dataset HYBRID -lifelong_name double_tokened -using_token_distillation_loss True -use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -learning_rate 0.001 -patience 3 -token_distillation_weight 10000 3000 800 200000 -token_num 10 -div_loss_weight 0.2
```
#### RELATION

```shell
python main.py -dataset RELATION -lifelong_name double_tokened -using_token_distillation_loss True -use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -learning_rate 0.001 -patience 3 -token_distillation_weight 3000 15000 80000 80000 -token_num 2 -div_loss_weight 0.2
```
#### WN-CKGE

```shell
python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True -use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -learning_rate 0.001 -patience 3 -token_distillation_weight 10000 5000 10000 10000 -token_num 10 -div_loss_weight 0.4
```
#### FB-CKGAE

```shell
python main.py -dataset WN_CKGE -lifelong_name double_tokened -using_token_distillation_loss True -use_multi_layers False -without_multi_layers True -use_two_stage False -batch_size 3072 -learning_rate 0.001 -patience 3 -token_distillation_weight 10000 10000 10000 20000 -token_num 10 -div_loss_weight 0.2
```



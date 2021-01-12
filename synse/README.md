### SynSE
Training and evaluation scripts for both SynSE and its gated variant have been provided.

The scripts accept the following arguments:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60); 24 (For NTU-120) | Which split to use |
phase | train; val | Which mode to run the model in |
ve | shift; msg3d | Select the Visual Embedding Model |
le | w2v; bert; bert_large | Select the Language Embedding Model |
dataset | - | Path to the NTU dataset |
wdir | - | Directory to store the weights to |

### Examples: 

### 1. Train SynSE on the NTU-60 dataset (without gating):

<code> python synse/synse_training.py --ntu 60 --phase train --ss 5 --st r -- dataset_path 'dataset/shift_5_r/' --wdir 'trained_models/' --le bert -- ve shift </code>

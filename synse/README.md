### SynSE
Training and evaluation scripts for both SynSE and its gated variant have been provided.
 

### SynSE (without gating):
The code for SynSE (without gating) is present in [synse_training.py](synse_training.py). The script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60); 24 (For NTU-120) | Which split to use |
phase | train; val | Which mode to run the model in |
ve | vacnn; shift; msg3d | Select the Visual Embedding Model |
le | w2v; bert | Select the Language Embedding Model |
dataset | - | Path to the NTU dataset |
wdir | - | Directory to store the weights to |

#### For example, to train SynSE on the NTU-60 55/5 split: 
<code> python synse/synse_training.py --ntu 60 --phase train --ss 5 --st r -- dataset_path 'dataset/shift_5_r/' --wdir 'trained_models/' --le bert -- ve shift </code>

#### To evaluate performance on the NTU-60 55/5 split:
<code> python synse/synse_training.py --ntu 60 --phase val --ss 5 --st r -- dataset_path 'dataset/shift_5_r/' --wdir 'trained_models/' --le bert -- ve shift </code>

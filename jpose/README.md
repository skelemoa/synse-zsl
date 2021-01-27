### JPOSE(FGAR)
Training and evaluation scripts for FGAR and the gating model have been provided.
 
### Training FGAR for ZSL:
1. The code for training FGAR for ZSL is present in [zsl_fgar.py](zsl_fgar.py). The script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60); 24 (For NTU-120) | Which split to use |
st | r (for random) | Split type |
phase | train; val | train(required for zsl), (once with train and once with val for gzsl) |
ve | shift; msg3d | Select the Visual Embedding Model |
le | w2v; bert | Select the Language Embedding Model |
dataset |- | Path to the generated visual features |
wdir | - | Path to the directory to store the weights in |
gpu | - | which gpu device number to train on |

2. Use the below command for training JPOSE for zsl for 55/5 split.
    <code> python zsl_fgar.py --ntu 60 --phase train --ss 5 --st r -- dataset 'synse_resources/ntu_results/shift_5_r/' --wdir 'synse_resources/language_modelling/fgar_5_r' --le bert -- ve shift --gpu 0</code>


### Train the gating model for FGAR:
1. The code to train FGAR (using gating) is present in [gating_fgar_model_training.py](gating_fgar_model_training.py). The script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60) | Which split to use |
phase | train; val | Which mode to run the model in |
ve | shift; msg3d | Select the Visual Embedding Model |
le | w2v; bert_large | Select the Language Embedding Model |
dataset |- | Path to the visual features |
wdir | - | Directory to store the weights in |

2. Training the gating model for 55/5 split:

    1. First train FGAR for ZSL in train phase:
    <code> python zsl_fgar.py --ntu 60 --phase train --ss 5 --st r -- dataset 'synse_resources/ntu_results/shift_5_r/' --wdir 'synse_resources/language_modelling/fgar_5_r' --le bert -- ve shift --gpu 0</code>
    2. Then train FGAR for ZSL in eval phase(Please note the difference in the wdir and dataset values):
    <code> python zsl_fgar.py --ntu 60 --phase val --ss 5 --st r -- dataset 'synse_resources/ntu_results/shift_val_5_r/' --wdir 'synse_resources/language_modelling/fgar_5_r' --le bert -- ve shift --gpu 0</code>
    3. Train the gating model using the following command.
    <code> python gating_fgar_model_training.py --ss 5 --st r --dataset 'synse_resources/ntu_results/shift_val_5_r/' --phase val --wdir synse_resources/language_modelling/fgar_5_r --le bert --ve shift --ntu 60 </code>
    4. Finally run the gating model eval script.
    <code> python gating_fgar_model_eval.py --ss 5 --st r --phase train --dataset synse_resources/ntu_results/shift_5_r --wdir synse_resources/language_modelling/fgar_5_r --le bert --ve shift --thresh 'Use value from output of previous step' --temp 'use value from output of previous step' --ntu 60 </code>


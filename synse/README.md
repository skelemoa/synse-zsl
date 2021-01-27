### SynSE
Training and evaluation scripts for SynSE and the gating model have been provided.
 

### Evaluate SynSE for ZSL using pretrained models:
1. The visual features can be found in the <code> synse_resources/ntu_results </code> directory. 
2. The pretrained models can be found in the <code> synse_resources/language_modelling </code> directory. 
3. For evaluating on the 5 random split of NTU-60
    <code> python synse_training.py --ntu 60 --phase train --ss 5 --st r -- dataset 'synse_resources/ntu_results/shift_5_r/' --wdir 'synse_resources/language_modelling/repo_test_synse_5_r' --le bert -- ve shift --load_classifier --num_cycles 10 --num_cycle_per_epoch 1700 --latent_size 100 --load_epoch 3399 --mode eval --gpu 0</code>
4. For evaluating on the 12 random split of NTU-60
    <code> python synse_training.py --ntu 60 --phase train --ss 12 --st r -- dataset 'synse_resources/ntu_results/shift_12_r/' --wdir 'synse_resources/language_modelling/repo_test_synse_12_r' --le bert -- ve shift --load_classifier --num_cycles 10 --num_cycle_per_epoch 1900 --latent_size 100 --load_epoch 6799 --mode eval --gpu 0</code>
5. For evaluating on the 10 random split of NTU-120
    <code> python synse/synse_training.py --ntu 120 --phase train --ss 10 --st r -- dataset 'synse_resources/ntu_results/shift_10_r/' --wdir 'synse_resources/language_modelling/repo_test_synse_10_r' --le bert -- ve shift --load_classifier --num_cycles 10 --num_cycle_per_epoch 1900 --latent_size 200 --load_epoch 7599 --mode eval --gpu 0</code>
6. For evaluating on the 24 random split of NTU-120
    <code> python synse/synse_training.py --ntu 120 --phase train --ss 24 --st r -- dataset 'synse_resources/ntu_results/shift_24_r/' --wdir 'synse_resources/language_modelling/repo_test_synse_24_r' --le bert -- ve shift --load_classifier --num_cycles 10 --num_cycle_per_epoch 1900 --latent_size 200 --load_epoch 9499 --mode eval --gpu 0</code>

### Training SYNSE for ZSL:
1. The code for training SynSE for ZSL is present in [synse_training.py](synse_training.py). The script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60); 24 (For NTU-120) | Which split to use |
st | r (for random) | Split type |
phase | train; val | train(required for zsl), (once with train and once with val for gzsl) |
ve | shift; msg3d | Select the Visual Embedding Model |
le | w2v; bert | Select the Language Embedding Model |
num_cycles | Integer | Number of cycles(Train for 10 cycles) |
num_epoch_per_cycle | Integer | Number of epochs per cycle 1700 for 5 random and 1900 for others|
latent_size | Integer | Size of the skeleton latent dimension (100 for ntu-60 and 200 for ntu-120)|
load_epoch | Integer | The epoch to be loaded |
load_classifier |  | Set if the pre-trained classifier is to be loaded |
dataset |- | Path to the generated visual features |
wdir | - | Path to the directory to store the weights in |
mode | train;eval | train for training synse, eval to eval using a pretrained model |
gpu | - | which gpu device number to train on |

2. Use the below command for training SYNSE for zsl for 55/5 split.
    <code> python synse_training.py --ntu 60 --phase train --ss 5 --st r -- dataset 'synse_resources/ntu_results/shift_5_r/' --wdir 'synse_resources/language_modelling/synse_5_r' --le bert -- ve shift --num_cycles 10 --num_cycle_per_epoch 1700 --latent_size 100 --mode train --gpu 0</code>



### Evaluate SynSE for GZSL using pretrained models:
1. For 55/5 split:
    <code> python gating_synse_model_eval.py --ss 5 --st r --phase train --dataset synse_resources/ntu_results/shift_5_r --wdir synse_resources/language_modelling/repo_test_synse_5_r --le bert --ve shift --thresh 0.54 --temp 3 --ntu 60 </code>
2. For 48/12 split:
    <code> python gating_synse_model_eval.py --ss 12 --st r --phase train --dataset synse_resources/ntu_results/shift_12_r --wdir synse_resources/language_modelling/repo_test_synse_12_r --le bert --ve shift --thresh 0.5 --temp 2 --ntu 60 </code>
3. For 110/10 split:
    <code> python gating_synse_model_eval.py --ss 10 --st r --phase train --dataset synse_resources/ntu_results/shift_10_r --wdir synse_resources/language_modelling/repo_test_synse_10_r --le bert --ve shift --thresh 0.5 --temp 3 --ntu 120 </code>
4. For 96/24 split:
    <code> python gating_synse_model_eval.py --ss 24 --st r --phase train --dataset synse_resources/ntu_results/shift_24_r --wdir synse_resources/language_modelling/repo_test_synse_24_r --le bert --ve shift --thresh 0.5 --temp 3 --ntu 120 </code>

### Train the gating model for Synse:
1. The code to train SynSE (using gating) is present in [gating_synse_model_training.py](gating_synse_model_training.py). The script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60) | Which split to use |
phase | train; val | Which mode to run the model in |
ve | shift; msg3d | Select the Visual Embedding Model |
le | w2v; bert_large | Select the Language Embedding Model |
dataset |- | Path to the visual features |
wdir | - | Directory to store the weights in |
thresh | - | Threshold for the binary classifier in the gating model |
temp | - | Temperature scaling factor in the gating model |

2. Training the gating model for 55/5 split:

To only train the gating model follow from step 3 with wdir value as 'synse_resources/language_modelling/repo_test_synse_5_r_val', otherwise continue from step 1.

    1. First train SYNSE for ZSL in train phase:
    <code> python synse_training.py --ntu 60 --phase train --ss 5 --st r -- dataset 'synse_resources/ntu_results/shift_5_r/' --wdir 'synse_resources/language_modelling/synse_5_r' --le bert -- ve shift --num_cycles 10 --num_cycle_per_epoch 1700 --latent_size 100 --mode train --gpu 0</code>
    2. Then train SYNSE for ZSL in eval phase(Please note the difference in the wdir and dataset values):
    <code> python synse_training.py --ntu 60 --phase val --ss 5 --st r -- dataset 'synse_resources/ntu_results/shift_val_5_r/' --wdir 'synse_resources/language_modelling/synse_5_r_val' --le bert -- ve shift --num_cycles 10 --num_cycle_per_epoch 1700 --latent_size 100 --mode train --gpu 0</code>
    3. Train the gating model using the following command.
    <code> python gating_synse_model_training.py --ss 5 --st r --dataset 'synse_resources/ntu_results/shift_val_5_r/' --phase val --wdir synse_resources/language_modelling/synse_5_r_val --le bert --ve shift --ntu 60 </code>
    4. Finally run the gating model eval script.
    <code> python gating_synse_model_eval.py --ss 5 --st r --phase train --dataset synse_resources/ntu_results/shift_5_r --wdir synse_resources/language_modelling/synse_5_r --le bert --ve shift --thresh 'Use value from output of previous step' --temp 'use value from output of previous step' --ntu 60 </code>


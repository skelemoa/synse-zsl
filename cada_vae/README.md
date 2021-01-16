### CADA-VAE
Training and evaluation scripts for both CADA-VAE and its gated variant have been provided.
 

### CADA-VAE (without gating):
The code for CADA-VAE (without gating) is present in [cada_vae_training.py](cada_vae_training.py). The script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60); 24 (For NTU-120) | Which split to use |
phase | train; val | Which mode to run the model in |
ve | vacnn; shift | Select the Visual Embedding Model |
le | w2v; bert | Select the Language Embedding Model |
gpu | String | GPU Device Number |
dataset |- | Path to the NTU dataset |
wdir | - | Directory to store the weights to |

#### For example, to train SynSE on the NTU-60 55/5 split: 
<code> python cada_vae/cada_vae_training.py --ntu 60 --phase train --ss 5 --st r -- dataset_path 'dataset/shift_5_r/' --wdir 'trained_models/' --le bert -- ve shift </code>

#### To evaluate performance on the NTU-60 55/5 split:
<code> python cada_vae/cada_vae_training.py --ntu 60 --phase val --ss 5 --st r -- dataset_path 'dataset/shift_5_r/' --wdir 'trained_models/' --le bert -- ve shift </code>


### CADA-VAE (with gating):
The code to train CADA-VAE (using gating) is present in [gating_cada_vae_model_training.py](gating_cada_vae_model_training.py). The script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
ntu | 60; 120 | Which NTU dataset to use |
ss | 5; 12 (For NTU-60) | Which split to use |
phase | train; val | Which mode to run the model in |
ve | vacnn; shift; msg3d | Select the Visual Embedding Model |
le | w2v; bert_large | Select the Language Embedding Model |
gpu | String | GPU Device Number |
dataset |- | Path to the NTU dataset |
wdir | - | Directory to store the weights to |

The code to evaluate CADA-VAE (using gating) is present in [gating_cada_vae_eval.py](gating_cada_vae_eval.py). <b> In addition to the arguments accepted by the training script </b>, this script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
temp | Integer | Temperature used for gating |
thresh | Float | The threshold used for gating |

#### For example, to train CADA-VAE (gating) on the NTU-60 55/5 split: 
<code> python cada_vae/gating_cada_vae_model_training.py  </code>

#### To evaluate performance (gating) on the NTU-60 55/5 split:
<code> python cada_vae/gating_cada_vae_model_eval.py  </code>

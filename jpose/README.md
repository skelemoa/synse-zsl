### JPoSE
Training and evaluation scripts for both JPoSE and its gated variant have been provided.
 

### JPoSE (without gating):
The code for JPoSE (without gating) is present in [zsl_revise.py](zsl_revise.py). The script accepts the following parameters:

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

#### For example, to train JPoSE on the NTU-60 55/5 split: 
<code> python revise/zsl_fgar.py --ntu 60 --phase train --ss 5 --st r -- dataset_path 'dataset/shift_5_r/' --wdir 'trained_models/' --le bert -- ve shift </code>

#### To evaluate performance on the NTU-60 55/5 split:
<code> python revise/zsl_fgar.py --ntu 60 --phase val --ss 5 --st r -- dataset_path 'dataset/shift_5_r/' --wdir 'trained_models/' --le bert -- ve shift </code>


### JPoSE (with gating):
The code to train JPoSE (using gating) is present in [gating_fgar_model_training.py](gating_fgar_model_training.py). The script accepts the following parameters:

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

The code to evaluate JPoSE (using gating) is present in [gating_fgar_eval.py](gating_fgar_eval.py). <b> In addition to the arguments accepted by the training script </b>, this script accepts the following parameters:

| Argument | Possible Values | Description |
--- | --- | --- | 
temp | Integer | Temperature used for gating |
thresh | Float | The threshold used for gating |

#### For example, to train JPoSE (gating) on the NTU-60 55/5 split: 
<code> python revise/gating_jpose_model_training.py  </code>

#### To evaluate performance (gating) on the NTU-60 55/5 split:
<code> python revise/gating_jpose_model_eval.py  </code>
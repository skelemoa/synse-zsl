# SynSE
Original PyTorch implementation for 'Syntactically Guided Generative Embeddings For Zero Shot Skeleton Action Recognition'

<img src = "Images/SynSE_arch.png" />

## Dependencies

<ul>
  <li> Python >= 3.6 </li>
  <li> Torchvision </li>
  <li> Scikit-Learn </li>  
</ul>

## Installation

### Download the NTU-60 and the NTU-120 datasets
  Download the NTU-60 and NTU-120 datasets by requesting them from <a href="http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp">here</a> .
  
### Preprocessing
 To generate the splits used by us, place the downloaded dataset in the 'dataset' folder and run the [split generator script](). The unseen classes of the various splits are listed below:
 
 
|Dataset | Split | Unseen Classes |
| --- | --- | --- | 
NTU-60 | 55/5 | Reading, Writing, Put on Hat or Cap, Jump Up, Touch Pocket |
NTU-60 | 48/12 |   |
NTU-120 | 96/24 |  |
 
### Setting up text embedding generators
<ol> 
  <li> Word2Vec: Download the <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit">Pre-Trained Word2Vec Vectors</a> and extract the contents of the archive </li>
  <li> For BERT, we use the sentence-transformers package. It can be installed using pip: 
    <br>
    <code> pip install -U sentence-transformers </code>
</ol>
  
## Experiments
We provide the scripts necessary to obtain the results shown in the paper. They include training and evaluation scripts for ReViSE (cite), JPoSE, CADA-VAE and our model SynSE.
The scripts for each of the three models are present in their respective folders (jpose, revise, synse). 
<br>
A README is present in each folder detailing the use of the provided scripts for both training and evaluation.

### todo
- [X] push code for synse
- [ ] push code for cadavae
- [X] push code for revise
- [X] push code for jpose
- [ ] push data preparation code
- [ ] push documentation
- [ ] provide resources and trained models for evaluation

link for trained models : https://drive.google.com/drive/folders/113edMAjmHlX9G81ToTX2AC74ejv7_2o8?usp=sharing

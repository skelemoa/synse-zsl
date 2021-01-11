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
 To generate the splits used by us, place the downloaded dataset in the 'dataset' folder and run the [split generator script]() .
 
### Setting up text embedding generators
<ol> 
  <li> Word2Vec: Download the <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit">Pre-Trained Word2Vec Vectors</a> and extract the contents of the archive </li>
  <li> For BERT, we use the sentence-transformers package. It can be installed using pip: 
    <br>
    <code> pip install -U sentence-transformers </code>
</ol>
  


###

### todo
- [X] push code for synse
- [ ] push code for cadavae
- [ ] push code for revise
- [ ] push code for jpose
- [ ] push data preparation code
- [ ] push documentation
- [ ] provide resources and trained models for evaluation

link for trained models : https://drive.google.com/drive/folders/113edMAjmHlX9G81ToTX2AC74ejv7_2o8?usp=sharing

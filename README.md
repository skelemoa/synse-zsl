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

### NTU-60: 
#### Unseen Classes (55/5 split):
<table>
  <tr>
    <td align = "center"><b>A11</b> reading </td>
    <td align = "center"><b>A12</b> writing </td>
    <td align = "center"><b>A20</b> put on a hat/cap </td>
    <td align = "center"><b>A27</b> jump up </td>
    <td align = "center"><b>A57</b> touch pocket </td>
  </tr>
</table>

#### Unseen Classes (48/12 split):
<table>
  <tr>
    <td align = "center"><b>A2</b> eat meal </td>
    <td align = "center"><b>A3</b> brush teeth </td>
    <td align = "center"><b>A15</b> take off jacket </td>
    <td align = "center"><b>A18</b> put on glasses </td>
    <td align = "center"><b>A21</b> take off a hat/cap </td>
  </tr>
  <tr>
    <td align = "center"><b>A25</b> reach into pocket </td>
    <td align = "center"><b>A32</b> taking a selfie </td>
    <td align = "center"><b>A38</b> salute </td>
    <td align = "center"><b>A40</b> cross hands in front </td>
    <td align = "center"><b>A44</b> headache </td>
  </tr>
  <tr>
    <td align = "center"><b>A47</b> neck pain </td>
    <td align = "center"><b>A57</b> touch pocket </td>
  </tr>
</table>

### NTU-120: 
#### Unseen Classes (110/10 split):
<table>
  <tr>
    <td align = "center"><b>A5</b> drop </td>
    <td align = "center"><b>A14</b> put on jacket </td>
    <td align = "center"><b>A38</b> salute </td>
    <td align = "center"><b>A44</b> headache </td>
    <td align = "center"><b>A50</b> punch or slap </td>
  </tr>
  <tr>
    <td align = "center"><b>A66</b> juggle table tennis table </td>
    <td align = "center"><b>A89</b> put object into bag </td>
    <td align = "center"><b>A96</b> cross arms </td>
    <td align = "center"><b>A100</b> butt kicks </td>
    <td align = "center"><b>A107</b> wield knife </td>
  </tr>
</table>

#### Unseen Classes (96/24 split):
<table>
  <tr>
    <td align = "center"><b>A2</b> eat meal </td>
    <td align = "center"><b>A4</b> brush hair </td>
    <td align = "center"><b>A5</b> drop </td>
    <td align = "center"><b>A11</b> reading </td>
    <td align = "center"><b>A18</b> put on glasses </td>
  </tr>
  <tr>
    <td align = "center"><b>A20</b> put on a hat/cap </td>
    <td align = "center"><b>A28</b> phone call </td>
    <td align = "center"><b>A48</b> nausea/vomiting </td>
    <td align = "center"><b>A49</b> fan self </td>
    <td align = "center"><b>A54</b> point finger </td>
  </tr>
   <tr>
    <td align = "center"><b>A59</b> walking towards </td>
    <td align = "center"><b>A64</b> bounce ball </td>
    <td align = "center"><b>A67</b> hush </td>
    <td align = "center"><b>A80</b> squat down </td>
    <td align = "center"><b>A84</b> play magic cube </td>
  </tr>
   <tr>
    <td align = "center"><b>A87</b> put on bag </td>
    <td align = "center"><b>A90</b> take object out of bag </td>
    <td align = "center"><b>A93</b> shake fist </td>
    <td align = "center"><b>A98</b> arm swings </td>
    <td align = "center"><b>A99</b> run on the spot </td>
  </tr>
  <tr>
    <td align = "center"><b>A101</b> cross toe touch </td>
    <td align = "center"><b>A104</b> stretch oneself </td>
    <td align = "center"><b>A106</b> hit with object </td>
    <td align = "center"><b>A109</b> grab stuff </td>
  </tr>
</table>

 
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
- [X] push code for cadavae
- [X] push code for revise
- [X] push code for jpose
- [ ] push data preparation code
- [ ] push documentation
- [X] provide trained models for evaluation
- [X] provide visual and language features for training and evaluation.

link for visual and language features and trained models : https://drive.google.com/file/d/167xoVJQ684XU1uFhSKD6j9nAwHsnmEky/view?usp=sharing

# Neural Image Caption Generation with Visual Attention

This repository contains a PyTorch implementation of the PMLR 2015 Paper, [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://proceedings.mlr.press/v37/xuc15.html). For the original implementation of the paper, please refer to the author's implementation in Theano language [here](https://github.com/kelvinxu/arctic-captions).

For further details on the results and analysis, please refer to. 


## Installing Dependencies

Make sure you have a Python3+ version. Run the following command - 

```
pip install -r requirements.txt
```

## Training the Model

### Dataset

Download the dataset from [here](https://drive.google.com/file/d/1h6Mg9RsjUIJyerOZlH4yYRQCIQ5GP0dZ/view?usp=sharing) and unzip the downloaded dataset. This will create a ```./data/``` in the root folder of the repository.  

### Model

The model definition is present in ```model.py``` and the training script is present in ```train.py```. 

## License 

Copyright (c) 2021  Paras Mehan

For license information, see [LICENSE](LICENSE) or http://mit-license.org


- - -

Done by [Pragyan Mehrotra](https://github.com/pragyanmehrotra), [Vrinda Narayan](https://github.com/vrindaaa) and Paras Mehan


This code was written as a part of a course group assignment in **Deep Learning** with [Dr. Md. Shad Akhtar](https://iiitd.ac.in/shad) at IIIT Delhi during Winter 2021 Semester.

For bugs in the code, please write to: paras18062 [at] iiitd [dot] ac [dot] in or create an issue in the repository

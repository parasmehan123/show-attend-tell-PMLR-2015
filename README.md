# Neural Image Caption Generation with Visual Attention

This repository contains a PyTorch implementation of the PMLR 2015 Paper, [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://proceedings.mlr.press/v37/xuc15.html). For the original implementation of the paper, please refer to the author's implementation in Theano language [here](https://github.com/kelvinxu/arctic-captions).

For further details on the results and analysis, please refer to. 


## Installing Dependencies

Make sure you have a Python3+ version. Run the following command - 

```bash
pip install -r requirements.txt
```

## Training the Model

### Dataset

Download the dataset from [here](https://drive.google.com/file/d/1h6Mg9RsjUIJyerOZlH4yYRQCIQ5GP0dZ/view?usp=sharing) and unzip the downloaded dataset. This will create a ```./data/``` in the root folder of the repository.  

### Training
```bash
python3 train.py [--base_dir (str)] [--debug (bool)] [--lr (str)] [--alpha_c (float)] [--log_interval (int)] [--epochs (int)] [--batch_size (int)] [--result_dir (str)] [--init_model (str)]
```
Options : 
```
--base_dir              Path of directory of data/ folder

--debug                 Debug : If set to True, then will print debug messages and will run only for one epoch with batch size one. 

--lr                    Learning Rate of Adam optimiser. 

--alpha_c               Regularisation Constant.

--log_interval          Number of batches after you want to print the loss for one epoch.

--epochs                No of epochs you want to train model. 

--batch_size            Batch Size to be used. 

--result_dir            Path of the directory where you want to create results/ folder which will save the trained models. 

--init_model            Path of the model with which you want to initialise the model before training. 
```

Example Usage : 
```bash
python3 train.py --base_dir=/kaggle/input/ass4q1/Data/ --result_dir=/kaggle/working/results --init_model=/kaggle/input/show-33/33.pth
```
## License 

Copyright (c) 2021  Paras Mehan

For license information, see [LICENSE](LICENSE) or http://mit-license.org


- - -

Done by [Pragyan Mehrotra](https://github.com/pragyanmehrotra), [Vrinda Narayan](https://github.com/vrindaaa) and Paras Mehan


This code was written as a part of a course group assignment in **Deep Learning** with [Dr. Md. Shad Akhtar](https://iiitd.ac.in/shad) at IIIT Delhi during Winter 2021 Semester.

For bugs in the code, please write to: paras18062 [at] iiitd [dot] ac [dot] in or create an issue in the repository

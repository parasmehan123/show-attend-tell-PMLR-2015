# Neural Image Caption Generation with Visual Attention

This repository contains a PyTorch implementation of the PMLR 2015 Paper, [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://proceedings.mlr.press/v37/xuc15.html). For the original implementation of the paper, please refer to the author's implementation in Theano language [here](https://github.com/kelvinxu/arctic-captions).

For further details on the Data Pre-Processing and Methodology, please refer to ```Report.pdf```

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
## Results
After training the model for 42 epochs, we got the following scores :

| Data  | BLEU1  | BLEU2  | BLEU3  | BLEU4  | METEOR  |
|---|---|---|---|---|---|
| Validation Data | 0.5560103298543799   | 0.3012887274731861   | 0.16869948803000315  |0.09124722446388864   | 0.24042306161781762  |
| Testing Data  | 0.5756577733000581  | 0.3240567011948614  | 0.18865787383495358  | 0.10922486278026644  |  0.25408193751079544  |

| Soft Attention for the generated images |
|----------------------|
| ![](Images/106490881_5a2dd9b7bd.jpg) |
| ![](Images/136644343_0e2b423829.jpg) |
| ![](Images/160792599_6a7ec52516.jpg) |
| ![](Images/249394748_2e4acfbbb5.jpg) |


## Inference

Download the trained model from [here](https://drive.google.com/file/d/1cCsgWKPzt7_ihJeUVwE3miNvCDmfmvT1/view?usp=sharing)

To finally test your trained model run the following command

```
python3 inferences.py [--base_dir (str)] [--model (str)] [--result_dir (int)] 
```
Options : 
```
--model                 Path of trained model

--base_dir              Path of directory of data/ folder

--result_dir            Path of the directory where you want to save the generated captions

```


## License 

Copyright (c) 2021  Paras Mehan

For license information, see [LICENSE](LICENSE) or http://mit-license.org


- - -

Done by [Pragyan Mehrotra](https://github.com/pragyanmehrotra), [Vrinda Narayan](https://github.com/vrindaaa) and Paras Mehan


This code was written as a part of a course group assignment in **Deep Learning** with [Dr. Md. Shad Akhtar](https://iiitd.ac.in/shad) at IIIT Delhi during Winter 2021 Semester.

For bugs in the code, please write to: paras18062 [at] iiitd [dot] ac [dot] in or create an issue in the repository

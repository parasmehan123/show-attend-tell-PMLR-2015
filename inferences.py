import argparse
import os
from model import show_attend_tell
from utils import dataset_helper
from pathlib import Path
from PIL import Image
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from math import ceil
import skimage
import skimage.transform
import matplotlib.cm as cm

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir',type=str,default="./")
parser.add_argument('--model',type=str,)
parser.add_argument('--result_dir',type=str,default="./results/")


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(os.path.join(args.result_dir,"output")):
        os.makedirs(os.path.join(args.result_dir,"output"))
        
    train_img_paths, train_captions, validation_img_paths, validation_captions, test_img_paths, test_captions, word_dict, idx_dict = dataset_helper(args.base_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = show_attend_tell(len(word_dict), 512, True,False)
    model.load_state_dict(torch.load(args.model))
    model = model.to(device)
    model.eval()
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    
    for img_path in tqdm(Path(f"{args.base_dir}/Data/Test/Images/").glob("*.jpg")):
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        img = val_transforms(img)
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)

        
        preds, alphas = model(img.to(device), 15) 
        packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
        sentence_tokens = []
        word_idxs = torch.max(preds, dim=2)[1]
        for idxs in word_idxs.tolist():
            for idx in idxs:
                if idx == word_dict['<eos>']:
                    break
                if idx != word_dict['<start>'] and idx != word_dict['<pad>']:
                    sentence_tokens.append(idx_dict[idx]) 

        img = Image.open(img_path)
        w, h = img.size
        if w > h:
            w = w * 256 / h
            h = 256
        else:
            h = h * 256 / w
            w = 256
        left = (w - 224) / 2
        top = (h - 224) / 2
        resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
        img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
        img = img.astype('float32') / 255

        num_words = len(sentence_tokens)
        w = np.round(np.sqrt(num_words))
        h = np.ceil(np.float32(num_words) / w)
        alpha = torch.tensor(alphas)

        plot_height = ceil((num_words + 3) / 4.0)
        plt.figure()
        ax1 = plt.subplot(4, plot_height, 1)
        plt.imshow(img)
        plt.axis('off')
        for idx in range(num_words):
            ax2 = plt.subplot(4, plot_height, idx + 2)
            label = sentence_tokens[idx]
            plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
            plt.text(0, 1, label, color='black', fontsize=13)
            plt.imshow(img)    
            shape_size = 14
            alpha_img = skimage.transform.pyramid_expand(alpha[0,idx,:].cpu().reshape(shape_size, shape_size), upscale=16, sigma=20)
            plt.imshow(alpha_img, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        plt.savefig(os.path.join(args.result_dir,"output",str(img_path).split('/')[-1]))
        plt.close()

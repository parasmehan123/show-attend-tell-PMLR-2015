from pathlib import Path
from nltk.tokenize import RegexpTokenizer
import pickle
from collections import Counter
from tqdm.auto import tqdm
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor_score
import numpy as np

def tokenize_gen_word_count(captions,name,tokenizer,word_count,max_length,base_dir = Path("./"), update = False):
    caption_tokens = []
    imgs_paths = []
    for img in captions:
        tmp = []
        for sentence in captions[img]:
            tokens = tokenizer.tokenize(sentence)
            tmp.append(tokens)
            if update:
                max_length = max(max_length, len(tokens))
                word_count.update(tokens)

        caption_tokens.append(tmp)
        imgs_paths.append(base_dir/f"Data/{name}/Images/{img}")
    return imgs_paths, caption_tokens, word_count,max_length

def get_tokes_from_captions(tokens,word_dict, max_length):
    captions = []
    for img_tokens in tokens:
        tmp = []
        for tk in img_tokens:

            token_idxs = [word_dict[token] if token in word_dict else word_dict['<unk>'] for token in tk]
            generated = [word_dict['<start>']] +  token_idxs + [word_dict['<eos>']] + [word_dict['<pad>']] * (max_length - len(tk))
            tmp.append(generated[:])
        captions.append(tmp)
    return captions

def dataset_helper(base_dir):
    base_dir = Path(base_dir)
    word_count = Counter()
    tokenizer = RegexpTokenizer(r'\w+')
    max_length = 0
    with open(base_dir/"Data/Train/train_captions.pkl","rb") as f:
        train_captions = pickle.load(f)
    with open(base_dir/"Data/Val/val_captions.pkl","rb") as f:
        val_captions = pickle.load(f)
    with open(base_dir/"Data/Test/test_captions.pkl","rb") as f:
        test_captions = pickle.load(f)


    train_img_paths, train_caption_tokens,word_count,max_length = tokenize_gen_word_count(train_captions, 'Train',tokenizer,word_count,max_length,base_dir,update=True)
    validation_img_paths, validation_caption_tokens,word_count,max_length = tokenize_gen_word_count(val_captions, 'Val',tokenizer,word_count,max_length,base_dir)
    test_img_paths, test_caption_tokens,word_count,max_length = tokenize_gen_word_count(test_captions, 'Test',tokenizer,word_count,max_length,base_dir)


    word_dict = {word: idx + 4 for idx, word in enumerate(list(word_count.keys()))}
    word_dict['<start>'] = 0
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2 #unkown
    word_dict['<pad>'] = 3

    idx_dict = {idx + 4: word  for idx, word in enumerate(list(word_count.keys()))}
    idx_dict[0] = '<start>'
    idx_dict[1] = '<eos>'
    idx_dict[2] = '<unk>'
    idx_dict[3] = '<pad>'

    train_captions = get_tokes_from_captions(train_caption_tokens,word_dict, max_length)
    validation_captions = get_tokes_from_captions(validation_caption_tokens,word_dict, max_length)
    test_captions = get_tokes_from_captions(test_caption_tokens,word_dict, max_length)


    return train_img_paths, train_captions, validation_img_paths, validation_captions, test_img_paths, test_captions, word_dict, idx_dict

def get_scores(model,loader,word_dict,idx_dict,device,debug):
    model.eval()
    references = []
    hypotheses = []
    for batch_idx, (imgs, captions,all_captions) in tqdm(enumerate(loader)):
        imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)
        max_timespan = max([len(caption) for caption in captions]) - 1 # -1, because assuming ke model already generated start token 
        preds, alphas = model(imgs, max_timespan) 
        
        for cap_set in all_captions.tolist():
            caps = []
            for caption in cap_set:
                cap = [word_idx for word_idx in caption
                                if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']]
                caps.append(cap)
            references.append(caps)

        word_idxs = torch.max(preds, dim=2)[1]
        for idxs in word_idxs.tolist():
            hypotheses.append([idx for idx in idxs
                                   if idx != word_dict['<start>'] and idx != word_dict['<pad>']])
        if debug:
            break
        
        
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses)
    score = []
    for i in range(len(references)):
        references_i = []
        for j in references[i]:
            words = []
            for k in j:
                words.append(idx_dict[k])
            references_i.append(' '.join(words))
        hypo_i = []
        for j in hypotheses[i]:
            hypo_i.append(idx_dict[j])
        score.append(meteor_score.meteor_score(references_i,' '.join(hypo_i)))
    return (bleu_1,bleu_2,bleu_3,bleu_4,np.mean(score))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_caption_lengths(word_dict, captions):
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths




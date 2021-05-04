from pathlib import Path
from nltk.tokenize import RegexpTokenizer
import pickle
from collections import Counter

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

def dataset(base_dir=Path("./")):

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


    return train_img_paths, train_captions, validation_img_paths, validation_captions, test_img_paths, test_captions



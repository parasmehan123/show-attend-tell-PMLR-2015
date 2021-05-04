import argparse
from utils import dataset_helper,AverageMeter,calculate_caption_lengths,get_scores
from dataset import ImageCaptionDataset
from model import show_attend_tell
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir',type=str,default="./")
parser.add_argument('--debug',type=bool,default=False)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--alpha_c',type=float,default=2.0)
parser.add_argument('--log_interval',type=int,default=10)
parser.add_argument('--epochs',type=int,default=1000)
parser.add_argument('--batch_size',type=int,default=64)
parser.add_argument('--result_dir',type=str,default="./results/")
parser.add_argument('--init_model',type=str,default="")



if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    debug = args.debug
    if debug:
        args.log_interval = 1
        args.epochs = 1
        args.batch_size = 1

    train_img_paths, train_captions, validation_img_paths, validation_captions, test_img_paths, test_captions, word_dict, idx_dict = dataset_helper(args.base_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = show_attend_tell(len(word_dict), 512, True,debug)
    if args.init_model !="":
        model.load_state_dict(torch.load(args.init_model))
    model = model.to(device)
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
    cross_entropy_loss = torch.nn.CrossEntropyLoss().to(device)

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    
    train_loader = torch.utils.data.DataLoader(
            ImageCaptionDataset(train_img_paths,train_captions, train_transforms),
            batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_loader = torch.utils.data.DataLoader(
            ImageCaptionDataset(validation_img_paths,validation_captions, val_transforms),
            batch_size=args.batch_size, shuffle=True, num_workers=1)

    test_loader = torch.utils.data.DataLoader(
            ImageCaptionDataset(test_img_paths,test_captions, val_transforms),
            batch_size=args.batch_size, shuffle=True, num_workers=1)

    for epoch in range(1, 1 + args.epochs):
        #scheduler.step()
        model.train()
        losses = AverageMeter()
        
        
        for batch_idx, (imgs, captions,all_captions) in enumerate(train_loader):
            imgs, captions = Variable(imgs).to(device), Variable(captions).to(device) # captions = (batch_size, max_len)
            if debug:
                print(f"imgs = {imgs.shape} captions = {captions.shape}")       
            
            optimizer.zero_grad()
            max_timespan = max([len(caption) for caption in captions]) - 1 # -1, because assuming ke model already generated start token 
            preds, alphas = model(imgs, max_timespan) 
            
            if debug:
                print(f"preds = {preds.shape} alphas = {alphas.shape}")
        
            targets = captions[:, 1:] # removing the start token 

            targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

            att_regularization = args.alpha_c * ((1 - alphas.sum(1))**2).mean()

            loss = cross_entropy_loss(packed_preds, targets)
            loss += att_regularization
            loss.backward()
            optimizer.step()

            total_caption_length = calculate_caption_lengths(word_dict, captions)
            losses.update(loss.item(), total_caption_length)
            if batch_idx % args.log_interval == 0:
                print('Train Batch: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        batch_idx, len(train_loader), loss=losses))
            if debug:
                break
        
        # x = get_scores(model,train_loader)
        y = get_scores(model,val_loader,word_dict,idx_dict,device,debug)
        z = get_scores(model,test_loader,word_dict,idx_dict,device,debug)
        torch.save(model.state_dict(),Path(args.result_dir)/f"{epoch}.pth")
        print(f"epoch = {epoch} Val : {y} Test : {z}")




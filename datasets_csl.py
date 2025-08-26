import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from ctypes import util
import utils as utils
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import cv2
from vidaug import augmentors as va
import clip

from definition import *

import random
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# Datasets for CSL-Daily

class S2T_CSLDataset(Dataset):
    def __init__(self,path,tokenizer,config,args,phase):
        self.config = config
        self.args = args
        
        self.raw_data = utils.load_dataset_file(path[phase])['info']
        self.tokenizer = tokenizer
        self.phase = phase
        self.text_feat = torch.load(config['data']['text_feat_path'][phase])
        self.max_length = config['data']['max_length']
        self.img_path = config['data']['img_path']
        
        self.list = [i for i in self.raw_data]   
       
        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            sometimes(va.RandomTranslate(x=10, y=10))
            # sometimes(va.Brightness(min=0.1, max=1.5)),
            # sometimes(va.Color(min=0.1, max=1.5)),
        ])

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index):
        sample = self.raw_data[index]
        
        tgt_sample = ''.join(sample['label_word'])
        name_sample = sample['name']
        text_sample = self.text_feat[name_sample]['bert_feat']
        
        folder_path = os.path.join(self.img_path, self.phase, name_sample)
        img_paths = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png') or file.endswith('.jpg')])
        img_sample = self.load_imgs(img_paths)

        return name_sample, text_sample, tgt_sample, img_sample
    
    def load_imgs(self, paths):

        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        if len(paths) > self.max_length:
            tmp = sorted(random.sample(range(len(paths)), k=self.max_length))
            new_paths = []
            for i in tmp:
                new_paths.append(paths[i])
            paths = new_paths
    
        imgs = torch.zeros(len(paths),3, self.args.input_size,self.args.input_size)
        crop_rect, resize = utils.data_augmentation(resize=(self.args.resize, self.args.resize), crop_size=self.args.input_size, is_train=(self.phase=='train'))

        batch_image = []
        for i,img_path in enumerate(paths):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_image.append(img)
            
        if self.phase == 'train':
            batch_image = self.seq(batch_image)
        
        for i, img in enumerate(batch_image):
            img_resized = img.resize(resize)
            img_tensor = data_transform(img_resized).unsqueeze(0)
            imgs[i,:,:,:] = img_tensor[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]

        return imgs

    def collate_fn(self,batch):
        
        tgt_batch,txt_tmp,src_length_batch,name_batch,img_tmp = [],[],[],[],[]

        for name_sample, txt_sample, tgt_sample, img_sample in batch:

            name_batch.append(name_sample)

            txt_tmp.append(txt_sample)

            tgt_batch.append(tgt_sample)
            
            img_tmp.append(img_sample)
                    
        max_len = max([len(vid) for vid in txt_tmp])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in txt_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad
        
        padded_txt = []
        
        for txt in txt_tmp:
            left_padded = txt[0].unsqueeze(0).expand(left_pad, -1)
            right_pad_len = max_len - len(txt) - left_pad
            right_padded = txt[-1].unsqueeze(0).expand(right_pad_len, -1)
            padded_tmp = torch.cat((left_padded, txt, right_padded), dim=0)
            padded_txt.append(padded_tmp)
            
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
            )
            , dim=0)
            for vid in img_tmp]

        
        img_tmp = [padded_video[i][0:video_length[i],:,:,:] for i in range(len(padded_video))]   
        
        new_txt_tmp = [padded_txt[i][0:video_length[i],:] for i in range(len(padded_txt))]
        
        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))
            
        src_length_batch = torch.tensor(src_length_batch)
        
        txt_batch = torch.cat(new_txt_tmp,0)
        img_batch = torch.cat(img_tmp, 0)

        new_src_lengths = (((src_length_batch-5+1) / 2)-5+1)/2
        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()
                
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt",padding = True,  truncation=True)

        src_input = {}
        src_input['input_txt'] = txt_batch
        src_input['input_img'] = img_batch
        src_input['attention_mask'] = img_padding_mask
        src_input['name_batch'] = name_batch

        src_input['src_length_batch'] = src_length_batch
        src_input['new_src_length_batch'] = new_src_lengths
                
        return src_input, tgt_input

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
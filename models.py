from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask
import torchvision
from torch.nn.utils.rnn import pad_sequence
#import pytorchvideo.models.x3d as x3d
import utils as utils
from peft import LoraConfig, get_peft_model, TaskType
from llavaov import LLaVA

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, MBart50Tokenizer, BertTokenizer, BertModel
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from PIL import Image
from collections import OrderedDict

import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from hpman.m import _
from pathlib import Path

def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    #model.fc = nn.Linear(inchannel, 768)
    return model

def to_btc(x, lengths):
    
    x_batch = []
    start = 0
    for length in lengths:
        end = start + length
        x_batch.append(x[start:end])
        start = end
    x = pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)
    
    return x

class resnet(nn.Module):
    def __init__(self, frozen=False):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

        if frozen:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)
        return x
        #return x_batch

    
class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)    


class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class MMSLT(nn.Module):
    def __init__(self, config, args, inplanes=768, planes=1024, pretrain=None,):
        super(MMSLT, self).__init__()
        self.config = config
        self.args = args
        
        model_id = "facebook/mbart-large-50-many-to-many-mmt"
        self.mbart = MBartForConditionalGeneration.from_pretrained(model_id, device_map=None)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # Apply LoRA to query and value layers
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        self.mbart = get_peft_model(self.mbart, lora_config)
        
        self.backbone =  resnet()
        # Description mapper
        self.descriptproj = Projector(input_dim=512, hidden_dim=planes, output_dim=inplanes)
        # Modality adapter
        self.conv = TemporalConv(input_size=512+inplanes, hidden_size=planes, conv_type=2)
        self.projector = Projector(input_dim=planes, hidden_dim=planes, output_dim=planes)
        # Freeze DM
        for param in self.descriptproj.parameters():
            param.requires_grad = False 

    def share_forward(self, src_input):
        
        img_feature = self.backbone(src_input['input_img'].cuda(), src_input['src_length_batch'])
        descript_feature = self.descriptproj(img_feature)
        inputs_embeds = torch.cat([img_feature, descript_feature], dim=-1)
        inputs_embeds = self.conv(inputs_embeds)
        inputs_embeds = self.projector(inputs_embeds)
        
        attention_mask = src_input['attention_mask']
                
        return inputs_embeds, attention_mask
        
    def forward(self, src_input, tgt_input):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)
                
        out = self.mbart(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask,
                    # decoder_input_ids = tgt_input['input_ids'].cuda(),
                    labels = tgt_input['input_ids'].cuda(),
                    decoder_attention_mask = tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        return out['logits']
    
    def generate(self,src_input,max_new_tokens,num_beams,forced_bos_token_id):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),
                    max_new_tokens=max_new_tokens,num_beams = num_beams,
                                forced_bos_token_id=forced_bos_token_id
                            )
        return out


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()

        self.model_txt = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").get_encoder() 
        for param in self.model_txt.parameters():
            param.requires_grad = False

    def forward(self, tgt_input):
        with torch.no_grad():
            txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0]
        
        output = txt_logits.mean(dim=1) # [b, 1024]
        return output


class ImageEncoder(nn.Module):
    def __init__(self, inplanes=768, planes=1024, head_type='linear') :
        super(ImageEncoder, self).__init__()
            
        self.backbone =  resnet()
        # Description mapper
        self.descriptproj = Projector(input_dim=512, hidden_dim=planes, output_dim=inplanes)
        # Modality Adapter
        self.conv = TemporalConv(input_size=512+768, hidden_size=planes, conv_type=2)
        self.projector = Projector(input_dim=planes, hidden_dim=planes, output_dim=planes)
        # Multimodal encoder
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").get_encoder()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # Apply LoRA to query and value layers
            lora_dropout=0.1,
            bias="none",
            task_type="ENCODER"
        )
        self.trans_encoder = get_peft_model(self.trans_encoder, lora_config)
        
    def forward(self, src_input):
        
        tgt_descript = to_btc(src_input['input_descript'].cuda(), src_input['src_length_batch'])
        img_feature = self.backbone(src_input['input_img'].cuda(), src_input['src_length_batch'])
        descript_feature = self.descriptproj(img_feature)
        mse_loss = F.mse_loss(descript_feature, tgt_descript)
        
        inputs_embeds = torch.cat([img_feature, descript_feature], dim=-1)
        inputs_embeds = self.conv(inputs_embeds)
        inputs_embeds = self.projector(inputs_embeds)

        attention_mask = src_input['attention_mask']
        
        outs = self.trans_encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        #output = last_hidden_state[:, 0, :] #[b, 1024]
        output = last_hidden_state.mean(dim=1)
                
        return output, mse_loss

class MMLP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(MMLP, self).__init__()
        self.model_text = TextEncoder()
        self.model_image = ImageEncoder(inplanes=768, planes=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, src_input, tgt_input):
        text_features = self.model_text(tgt_input)
        image_features, descript_loss = self.model_image(src_input)

        # normalized features
        norm_text = F.normalize(text_features, p=2, dim=-1)
        norm_images = F.normalize(image_features, p=2, dim=-1)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        sim_text = torch.matmul(norm_text, norm_images.t()) * logit_scale
        sim_image = torch.matmul(norm_images, norm_text.t()) * logit_scale

        return sim_text, sim_image, descript_loss
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import numpy as np
import os
import argparse
import sys
import signal

from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from datasets import VideoDataset, MissDataset
from llavaov import LLaVA
from collections import defaultdict

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_feature(args):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    batch_size = args.video_bs
    frame_batch = args.frame_bs
    resume = args.resume
    split = args.split
    save_path = args.save_path
    
    if resume:
        dataset = MissDataset(vars(args),split)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

        dataloader = DataLoader(
                dataset=dataset, 
                batch_size=batch_size, 
                #shuffle=True,
                sampler=sampler,
                drop_last=False,
                num_workers=16,
                )
        
    else:
        dataset = VideoDataset(vars(args),split)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

        dataloader = DataLoader(
                dataset=dataset, 
                batch_size=batch_size, 
                #shuffle=True,
                sampler=sampler,
                drop_last=False,
                num_workers=16,
                )
    
    mmlm = LLaVA()
    text_dict = defaultdict(dict)
    cnt = 0

    for batch in tqdm(dataloader):
        #vid_name, images, image_sizes = batch
        vid_name, image_paths = batch
        images = [Image.open(f[0]).convert("RGB") for f in image_paths] # with CSL-Daily, add .resize((256,256))
        vid_name = vid_name[0]
        idx = [0]
        for i in range(0, len(images), frame_batch):
            slice = min(i + frame_batch, len(images))
            idx.append(slice)

        vid_texts = []
        #vid_lengths = []
        for i in range(len(idx)-1):
            start, end = idx[i:i+2]
            frames = images[start:end]
            
            texts = mmlm(images=frames)
            vid_texts.extend(texts)
            #vid_lengths.extend(lengths)
        
        text_dict[vid_name]['texts'] = vid_texts
       # text_dict[vid_name]['text_lengths'] = vid_lengths
                            
        cnt += 1
                
        if cnt % 100 == 0: # save codebook in every 100 iteration (backup)
            if resume:
                torch.save(text_dict, f'{save_path}phoenix_SLdescript.{split}_{rank}') # train, dev, test
            else:
                torch.save(text_dict, f'{save_path}phoenix_SLdescript.{split}_{rank}') # train, dev, test
                
            print(f"Saving features in {cnt} iteraion..")
                    
    if resume:
        torch.save(text_dict, f'{save_path}phoenix_SLdescript.{split}_{rank}')
    else:
        torch.save(text_dict, f'{save_path}phoenix_SLdescript.{split}_{rank}')
        
    print("Saving features complete!")
        
def cleanup():
    # Any cleanup code, e.g., free up CUDA memory
    torch.cuda.empty_cache()
    sys.exit(0)
    
def signal_handler(sig, frame):
    cleanup()
    
signal.signal(signal.SIGINT, signal_handler)

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--ngpus', type=int, default=1, help='number of gpus used')
    #parser.add_argument('--local_rank', type=int, default=0, help='rank of the current process')
    parser.add_argument('--img_path', type=str, default='path/to/datasets/', # train, dev, test
                        help='path to dataset folder')
    parser.add_argument('--split', type=str, default='train', help='split')
    parser.add_argument('--frame_bs', type=int, default=8, help='batch size of frames for LLaVA input')
    parser.add_argument('--video_bs', type=int, default=1)
    parser.add_argument('--resume', type=bool, default=False, help='resume generating features')
    parser.add_argument('--save_path', type=str, default= "path/to/save/", help='path to save features')
    args = parser.parse_args()

    create_feature(args)    
    
if __name__ == "__main__":
    main()

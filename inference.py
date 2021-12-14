#!/usr/bin/env python3
import os
import time
import argparse
import logging
import numpy as np
import torch
from PIL import Image
from timm.models import create_model, apply_test_time_pool, load_checkpoint
from timm.data import ImageDataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from timm.data.transforms_factory import create_transform

def predict_one_img(model,device,config,img_data,k,):
    print("[output]------------------------------------------------------")
    print(config)
    transform=None
    if config is None:
        print("change")
        config=224
        transform = create_transform(input_size=config)
    else:
        transform = create_transform(**config)
    class_map = ['充电车位', '子母车位', '子车位', '微型充电车位', '微型车位', '无障碍车位', '机械车位', '货车车位', '车位', 'out of range']
    

    print("[transform]------------------------------------------------------")
    # filename = os.path.join(root_path,img_path)
    # img = Image.open(filename).convert('RGB')
    img_tensor = transform(img_data).unsqueeze(0)  # transform and add batch dimension
    img_tensor = img_tensor.to(device)
    
    img_label = model(img_tensor)
    topk = img_label.topk(k)[1]
    topk_ids = topk.cpu().numpy()
    topk_ids = np.concatenate(topk_ids, axis=0)

    json_data=[]
    for i in range(k):
        tmp_id = topk_ids[i] if topk_ids[i]<9 else 9
        r = class_map[tmp_id]
        json_data.append(r)

    return json_data

def prepare(checkpoint_path,topk):
    # setup_default_logging()
    # args = parser.parse_args()
    # might as well try to do something useful...

    device = "cpu"

    # create model
    model = create_model(
        'efficientnet_b0',
        num_classes=1000,
        in_chans=3,
        pretrained=False,
        checkpoint_path=checkpoint_path)

    config = resolve_data_config({}, model=model)
    model = model.to(device)
    print("[config]------------------------------------------------------")
    if config is None:
        print("[inference]:config none")
    if model is None:
        print("[inference]:model none")
    model.eval()

    k = min(topk, 1000)

    return model,config
    # if args.img_name:
    #     predict_one_img(model=model,device=device,config=config,img_path=args.img_name,root_path=args.data,k=k)
    # else :
    #     predict_one_folder(model=model,device=device,data_path=args.data,output_dir=args.output_dir,config=config,
    #                        batchsize=args.batch_size,workers=args.workers,k=k,test_time_pool=test_time_pool,log_freq=args.log_freq)
    #

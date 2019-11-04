#!/usr/bin/env python
# coding: utf-8

# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import time
import helper
from PIL import Image
import numpy as np
import os
import argparse


def load_checkpoint(filepath):
    ''' Restores a Checkpoint'''
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    checkpoint = torch.load(filepath, map_location=map_location)
    return checkpoint

def load_model(checkpoint):
    ''' Restores a Model from a saved checkpoint'''
    model_new = models.vgg16(pretrained=True)
    model_new.classifier = checkpoint['classifier']
    model_new.load_state_dict(checkpoint['state_dict_model'])
    model_new.class_to_idx = checkpoint['class_to_idx']    
    return model_new

def load_optimizer(checkpoint, model_new):
    ''' Restores a Optimizer from a saved checkpoint'''
    optimizer_new = optim.Adam(model_new.classifier.parameters())
    optimizer_new.load_state_dict(checkpoint['state_dict_optimizer'])
    return optimizer_new

def process_image(image_pil):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    size = 256, 256
    
    # Resize Image
    image_pil.thumbnail(size, Image.ANTIALIAS)
    
    # Crop center of the image
    width, height = image_pil.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image_pil = image_pil.crop((left, top, right, bottom))
    
    # Convert integers 0-255 to float 0-1
    image_np = np.array(image_pil)/255
    
    # Normalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    # Transpose array to switch colour channel's dimension from 2 --> 0
    image_np = image_np.transpose((2, 0, 1))
    return image_np    

def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    img_pil = Image.open(image_path, 'r')
    img_np = process_image(img_pil)
    
    checkpoint = load_checkpoint(model)
    model_new = load_model(checkpoint)
    optimizer_new = load_optimizer(checkpoint, model_new)

    # Add batch dimension
    img_tensor = torch.from_numpy(img_np)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()

    # Determine Default Device
    if gpu in ['Y', 'y']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print("GPU unavailable, switching to CPU...")
    else:
        device = "cpu"

    print("Default Device:", device)
    model_new.to(device)
    
    img_tensor = img_tensor.to(device)
    
    # Predict
    model_new.eval()

    with torch.no_grad():
            logps = model_new.forward(img_tensor)
            ps = torch.exp(logps)
            probs, indexes = ps.topk(topk, dim=1)

    model_new.train() 
    
    cls2idx = model_new.class_to_idx
    idx2cls = {v: k for k, v in cls2idx.items()}
    
    # As cuda does not support numpy(), this operation is always performed on cpu
    indexes_np = indexes.cpu().numpy()
    
    classes = []
    for index in zip(indexes_np[0]):
        classes.append(idx2cls[index[0]])
    
    return probs, classes

# Define Argument Parser
parser = argparse.ArgumentParser(description='Predict Parameters')
parser.add_argument('image_path', action="store", help='Image Path')
parser.add_argument('checkpoint', action="store", help='Checkpoint Location')
parser.add_argument('--top_k', action="store", dest='top_k', type=int, help='Top K')
parser.add_argument('--category_names', action="store", dest='category_names', help='Category Names')
parser.add_argument('--gpu', action="store", dest='gpu', help='Use GPU [Y/N]')
args = parser.parse_args()

im = Image.open(args.image_path, 'r')
img_proc = process_image(im)

if args.top_k is None:
    probs, classes = predict(args.image_path, args.checkpoint, args.gpu)
else:
    probs, classes = predict(args.image_path, args.checkpoint, args.gpu, args.top_k)

# Display Category Names if opted for, else display class
if args.category_names is None:
    for a, b in zip(probs[0].numpy(), classes):
        print(a, ' --> ', b)
else:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Retrieve flower names
    flowers = []
    for flower in zip(classes):
        flowers.append(cat_to_name[flower[0]])
    
    for a, b in zip(probs[0].numpy(), flowers):
        print(a, ' --> ', b)

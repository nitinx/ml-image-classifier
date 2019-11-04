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
from workspace_utils import active_session
from PIL import Image
import numpy as np
import os
import argparse

# Define Argument Parsers
parser = argparse.ArgumentParser(description='Train Parameters')
parser.add_argument('data_dir', action="store", help='Data Directory')
parser.add_argument('--save_dir', action="store", dest='save_dir', help='Checkpoint Save Directory')
parser.add_argument('--arch', action="store", dest='arch', help='Model Architecture')
parser.add_argument('--learning_rate', action="store", dest='learning_rate', type=float, help='Model Learning Rate')
parser.add_argument('--hidden_units', action="store", dest='hidden_units', type=int, help='Model Hidden Units')
parser.add_argument('--epochs', action="store", dest='epochs', type=int, help='Epochs')
parser.add_argument('--gpu', action="store", dest='gpu', help='Use GPU')
args = parser.parse_args()

# Define Data Directories
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# Define Transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), 
                                       normalize])

transforms_valid = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize])

# Define ImageFolders
data_train = datasets.ImageFolder(train_dir, transform=transforms_train)
data_valid = datasets.ImageFolder(valid_dir, transform=transforms_valid)

# Define Dataloaders
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(data_valid, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Determine Architecture
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
else:
    model = models.vgg19(pretrained=True)
print("Model:", model)

# Determine Hidden Units
if args.hidden_units is None:
    hidden_units = 1024
else:
    hidden_units = args.hidden_units
print("Hidden Units:", hidden_units)

# Freeze parameters so we dont backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 4096)),
    ('relu1', nn.ReLU()),
    ('dp1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(4096, hidden_units)),
    ('relu2', nn.ReLU()),
    ('dp2', nn.Dropout(0.2)),
    ('fc4', nn.Linear(hidden_units, len(cat_to_name))),
    ('output', nn.LogSoftmax(dim=1))
]))

# Determine Learning Rate
if args.learning_rate is None:
    lr = 0.0005
else:
    lr = args.learning_rate
print("Learning Rate:", lr)

model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# Determine Default Device
if args.gpu in ['Y', 'y'] and torch.cuda.is_available(): 
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Default Device:", device)
model.to(device)

# Determine Save Directory
if args.save_dir is None:
    save_dir = "chkpt.pth"
else:
    save_dir = args.epochs + "/chkpt.pth"
print("Save Directory:", save_dir)

# Determine Epochs
if args.epochs is None:
    epochs = 1
else:
    epochs = args.epochs
print("Epochs:", epochs)

# Train Model
steps = 0
print_every = 5
running_loss = 0

# Keep active long running sessions
with active_session():
    
    # Loop through epochs
    for epoch in range(epochs):

        for inputs, labels in dataloader_train:
            steps += 1

            # Move tensors to default device  
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute output
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            # Compute gradient and backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                
                model.eval()
                
                with torch.no_grad():
                
                    for inputs, labels in dataloader_valid:
                        
                        # Move tensors to default device
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        # Compute output
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Compute accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1/epochs}.. "
                      f"Train Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {test_loss/len(dataloader_valid):.3f}.. "
                      f"Accuracy: {accuracy/len(dataloader_valid):.3f}" )
                       
                running_loss = 0
                model.train()   

model.class_to_idx = data_train.class_to_idx

# Save Checkpoint
checkpoint = {'epoch': epoch,
              'classifier': model.classifier,
              'state_dict_model': model.state_dict(),
              'state_dict_optimizer': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx
             }

torch.save(checkpoint, 'chkpt.pth')

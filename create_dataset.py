'''
CS5100 Foundations of Artificial Intelligence
Project
Author: Arun Madhusudhanana, Tejaswini Dilip Deore

This script is used to create a dataset class for the Flickr8k dataset. The dataset class is used to create a train and test dataloader.

'''

# Importing the required libraries
import torch
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import h5py
import pandas as pd
import numpy as np
import os, json


class FlickrDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, image_dir, resnet_features, transform=None):
        '''
        This function is used to initialize the FlickrDataset class.
        Args:
            df (DataFrame): DataFrame containing the image names and captions
            tokenizer (Tokenizer): Tokenizer object
            image_dir (str): Path to the images folder
            resnet_features (dict): Dictionary containing the resnet features for each image
            transform (torchvision.transforms): Transform to be applied to the images
        '''
        self.df = df
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.transform = transform
        self.resnet_features = resnet_features
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        '''
        This function is used to get the item from the dataset.
        Args:
            idx (int): Index of the item to be fetched from the dataset.
        Returns:
            img (tensor): Tensor of shape (C, H, W)
        '''
        image_path = self.image_dir + '/' + self.df.iloc[idx, 0]
        img = Image.open(image_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)

        caption = self.df.iloc[idx, 1].lower()
        caption = "<SOS> " + caption + " <EOS>"
        tokens = self.tokenizer.encode(caption).ids
        caption_tokens = torch.tensor(tokens)
        
        res_features = self.resnet_features[self.df.iloc[idx, 0]]
        res_features = torch.tensor(res_features.squeeze())      

       
        return img, caption_tokens, self.df.iloc[idx, 0], res_features
    
def collate_fn(batch, pad_idx = 0, batch_first = True):
    '''
    This function is used to collate the batch of images and captions.
    Args:
        batch (list): List of tuples where each tuple is (image, caption, image_name, resnet_features)
        pad_idx (int): Padding index
        batch_first (bool): If True, the output will be of shape (batch_size, max_seq_len, ...)
                            If False, the output will be of shape (max_seq_len, batch_size, ...)

    Returns:
        images (tensor): Tensor of shape (batch_size, C, H, W)
        captions (tensor): Tensor of shape (batch_size, max_seq_len)
        image_names (list): List of image names
        resnet_features (tensor): Tensor of shape (batch_size, 512)
    '''
    images = [item[0].unsqueeze(0) for item in batch]
    images = torch.cat(images, dim=0)
    captions = [item[1] for item in batch]
    image_names = [item[2] for item in batch]
    resnet_features = [item[3] for item in batch]
    resnet_features = torch.stack(resnet_features)
    captions = pad_sequence(captions, batch_first=batch_first, padding_value=pad_idx)
    return images, captions, image_names, resnet_features


# def load_coco_data(base_dir,
#                    max_train=None,
#                    pca_features=True):
#     data = {}
#     caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
#     with h5py.File(caption_file, 'r') as f:
#         for k, v in f.items():
#             data[k] = np.asarray(v)

#     if pca_features:
#         train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
#     else:
#         train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
#     with h5py.File(train_feat_file, 'r') as f:
#         data['train_features'] = np.asarray(f['features'])

#     if pca_features:
#         val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
#     else:
#         val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
#     with h5py.File(val_feat_file, 'r') as f:
#         data['val_features'] = np.asarray(f['features'])

#     dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
#     with open(dict_file, 'r') as f:
#         dict_data = json.load(f)
#         for k, v in dict_data.items():
#             data[k] = v

#     train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
#     with open(train_url_file, 'r') as f:
#         train_urls = np.asarray([line.strip() for line in f])
#     data['train_urls'] = train_urls

#     val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
#     with open(val_url_file, 'r') as f:
#         val_urls = np.asarray([line.strip() for line in f])
#     data['val_urls'] = val_urls

#     # Maybe subsample the training data
#     if max_train is not None:
#         num_train = data['train_captions'].shape[0]
#         mask = np.random.randint(num_train, size=max_train)
#         data['train_captions'] = data['train_captions'][mask]
#         data['train_image_idxs'] = data['train_image_idxs'][mask]

#     return data

# class MSCOCODataset(torch.utils.Dataset):
#     def __init__(self, data):
#         self.data = data


        
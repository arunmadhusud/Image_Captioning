# Project Description

<!-- This repository contains the code for project 'Image Caption Generator' completed as the final project for course 'CS5100: Foundations of Artificial Intelligence'.  -->

Contributors : Arun Madhusudhanan, Tejaswini Dilip Deore.

Describing an image automatically by generating meaningful captions is a fundamental problem in Artificial Intelligence. Recent advancements in Large Language Models have significantly improved the performance of this task. Previously the most common approach involved using a convolutional neural network (CNN) as  encoder and a Recurrent neural network(RNN) as decoder for this task. In this project, we present an Image Caption Generator utilizing two distinct architectures: a Convolutional Neural Network(CNN) encoder with Long Short-term memory(LSTM) decoder, and a transformer-based model using Vision Transformer (ViT) and Generative Pre-trained Transformer 2 (GPT-2). We compare performance of these models qualitatively and quantitatively. Both architectures were evaluated using metrics like BLEU score, ROUGE score, METEOR score, and CIDEr score on Flickr8k dataset. Our findings indicate that both ViT-GPT2 model and CNN-LSTM model were able to generate meaningful and descriptive  captions for images.

The fine tuned ViT+GPT2 model is hosted on Hugging face platform which can be accesssed using the link https://huggingface.co/arunmadhusudh/Vit-gpt2-flickr8k


## Dataset used

Please refer to Jason Brownlee's GITHUB link to Download Flickr_8k dataset

1. Flickr8k_Dataset.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

2. Flickr8k_text.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

## Instructions

1. Run main.py to train and evaluate CNN-LSTM model. Set the correct directory to load image data and caption data.
2. Run ViT-GPT2.ipynb to fine tune ViT-GPT2 model.Set the correct directory to load image data and caption data.
3. Run ViT-GPT2 inference.ipynb to run inference using finetuned ViT-GPT2.Set the correct directory to load image data and caption data.


## Acknowledgements

1. Jason Brownlee : Hosting Flickr_8K dataset
2. Hugginface : Downloaded pretrained weights for ViT and GPT2
3. Ramakrishna Vedantam and Tsung-Yi Lin: Authors of cider.py and cider_score.py

'''
CS5100 Foundations of Artificial Intelligence
Project
Author: Arun Madhusudhanana, Tejaswini Dilip Deore

This file contains utility functions that are used in the training and testing of the model. 

'''

# Importing the required libraries
import evaluate
from tqdm import trange
from cider import Cider
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
import os


def get_or_build_tokenizer(df, tokenizer_path, vocab_size=5000):
    '''
    This function is used to get the tokenizer if it exists or build the tokenizer if it does not exist.
    Args: 
        df (DataFrame): The dataframe containing the captions
        tokenizer_path (str): The path to the tokenizer
        vocab_size (int): The size of the vocabulary
    Returns:
        tokenizer (Tokenizer) : The tokenizer object
    '''
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        with open('data.txt', 'w') as file:
            for caption in df['caption'].values: 
                file.write(caption.lower() + '\n')
        data = 'data.txt'
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>")) 
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<PAD>", "<SOS>", "<EOS>", "<UNK>"],min_frequency=5) 
        tokenizer.train([data], trainer)  # Pass data directly here
        tokenizer.save(tokenizer_path)
    return tokenizer



def display_image_caption(image):
    '''
    This function is used to display the image and the caption.
    Args:
        image (tensor) : The image tensor
    '''
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


def train_cnn_lstm_model(model, criterion, optimizer, train_loader, device, vocab_size):  
    '''
    This function is used to train the CNN-LSTM model
    Args:
        model (nn.Module) : The model
        criterion (nn.Module) : The loss function
        optimizer (optim) : The optimizer
        train_loader (DataLoader) : The training data loader
        device (str) : The device
        vocab_size (int) : The size of the vocabulary
    Returns:    
        train_loss (float) : The training loss
    '''
    model.train()
    train_loss = 0
    for i, (images, captions,_,image_features) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        image_features = image_features.to(device)   
        optimizer.zero_grad()
        outputs = model(image_features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss
       

def validate_cnn_lstm_model(model, criterion, test_loader, device, vocab_size):
    '''
    This function is used to validate the CNN-LSTM model
    Args:
        model (nn.Module) : The model
        criterion (nn.Module) : The loss function
        test_loader (DataLoader) : The test data loader
        device (str) : The device
        vocab_size (int) : The size of the vocabulary
    Returns:
        validation_loss (float) : The validation loss
    '''
    model.eval()
    with torch.no_grad():
        for i,(images, captions, image_names, image_features) in enumerate(test_loader):
            images = images.to(device)
            captions = captions.to(device)
            image_features = image_features.to(device)
            outputs = model(image_features, captions)
            validation_loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
    return validation_loss

def test_model(model, test_dataset, device, tokenizer):
    '''
    This function is used to test the model
    Args:
        model (nn.Module) : The model
        test_dataset (Dataset) : The test dataset
        device (str) : The device
        tokenizer (Tokenizer) : The tokenizer object
    '''
    model.eval()
    with torch.no_grad():
        idx = random.randint(0, len(test_dataset))
        # print(idx)
        image, caption, _, image_features = test_dataset[idx]
        image = image.unsqueeze(0).to(device)
        caption = caption.unsqueeze(0).to(device)
        image_features = image_features.unsqueeze(0).to(device)
        features = model.encoder(image_features).to(device)
        # print(features.shape)
        print("\nGround Truth: ", tokenizer.decode(caption[0].cpu().numpy()))
        predicted = model.decoder.sample_caption(features.unsqueeze(0), tokenizer = tokenizer)
        print("Greedy Search: ", " ".join(predicted))
        predicted_beam = model.decoder.beam_search(features, tokenizer = tokenizer, beam_size=3)
        print("Beam Search: ", " ".join(predicted_beam))
        display_image_caption(image[0].cpu())

def compute_scores(model, test_loader, device, tokenizer, df):
    '''
    This function is used to compute the BLEU, ROUGE, METEOR, CIDEr scores for the model
    Args:
        model (nn.Module) : The model
        test_loader (DataLoader) : The test data loader
        device (str) : The device
        tokenizer (Tokenizer) : The tokenizer object
        df (DataFrame) : The dataframe containing the captions
    Returns:
        bleu_1_greedy (float) : The BLEU-1 score for Greedy Search
        bleu_2_greedy (float) : The BLEU-2 score for Greedy Search
        rouge1_greedy (float) : The ROUGE-1 score for Greedy Search
        rougeL_greedy (float) : The ROUGE-L score for Greedy Search
        meteor_greedy (float) : The METEOR score for Greedy Search
        cider_greedy (float) : The CIDEr score for Greedy Search
        bleu_1_beam (float) : The BLEU-1 score for Beam Search
        bleu_2_beam (float) : The BLEU-2 score for Beam Search
        rouge1_beam (float) : The ROUGE-1 score for Beam Search
        rougeL_beam (float) : The ROUGE-L score for Beam Search
        meteor_beam (float) : The METEOR score for Beam Search
        cider_beam (float) : The CIDEr score for Beam Search
    '''
    model.eval()
    # Create a Cider object
    cider = Cider()

    bleu_metric = evaluate.load('bleu')
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')

    # BLEU, ROUGE, METEOR, CIDEr scores for Greedy Search
    bleu_1_greedy = []
    bleu_2_greedy = []
    rouge1_greedy = []
    rougeL_greedy = []
    meteor_greedy = []
    cider_greedy = []

    # BLEU, ROUGE, METEOR, CIDEr scores for Beam Search
    bleu_1_beam = []
    bleu_2_beam = []
    rouge1_beam = []
    rougeL_beam = []
    meteor_beam = []
    cider_beam = []

    model.eval()

    with torch.no_grad():
        images, captions, image_names, image_features = next(iter(test_loader))
        images = images.to(device)
        captions = captions.to(device)
        image_features = image_features.to(device)
        for k in range(images.shape[0]):
            predictions_greedy = []
            predictions_beam = []
            references = []
            cider_predictions_greedy = {'image1': ['This is a dummy data for cider.py to work']}
            cider_predictions_beam = {'image1': ['This is a dummy data for cider.py to work']}
            cider_references = {'image1': ['This is a dummy data for cider.py to work']}
            image = images[k].unsqueeze(0)
            caption = captions[k].unsqueeze(0)
            image_feature = image_features[k].unsqueeze(0)
            image_name = image_names[k]
            features = model.encoder(image_feature)

            # Greedy Search
            predicted_greedy = model.decoder.sample_caption(features.unsqueeze(0), tokenizer = tokenizer)
            predictions_greedy.append(predicted_greedy)

            # Beam Search
            predicted_beam = model.decoder.beam_search(features, tokenizer = tokenizer, beam_size=3)
            predictions_beam.append(predicted_beam)

            # True Captions
            idx = df[df['image'] == image_name].index
            true_captions = []
            for j in idx:
                true_captions.append(df.iloc[j, 1])
            references.append(true_captions)

            ############################################################################################

            ''' Compute BLEU, ROUGE, METEOR, CIDEr scores for Greedy Search and Beam Search'''

            # CIDEr predictions and references for greedy search
            cider_predictions_greedy[image_name] = [predicted_greedy]
            cider_references[image_name] = true_captions

            # compute cider score
            score,scores = cider.compute_score(cider_references, cider_predictions_greedy)
            # print(scores)
            cider_greedy.append(scores[1]) # index 0 corresponds to dummy data, so we take index 1
            
            # compute bleu
            bleu = bleu_metric.compute(predictions=predictions_greedy, references=references)
            bleu_1_greedy.append(bleu['precisions'][0])
            bleu_2_greedy.append(bleu['precisions'][1])
            
            # compute rouge
            rouge = rouge_metric.compute(predictions=predictions_greedy, references=references)
            rouge1_greedy.append(rouge['rouge1'])
            rougeL_greedy.append(rouge['rougeL'])
            
            # compute meteor
            meteor = meteor_metric.compute(predictions=predictions_greedy, references=references)
            meteor_greedy.append(meteor['meteor'])

            ##########################################################################################
            ''' Compute BLEU, ROUGE, METEOR, CIDEr scores for Beam Search'''

            # CIDEr predictions and references for beam search
            cider_predictions_beam[image_name] = [predicted_beam]

            # compute cider score
            beam_score,beam_scores = cider.compute_score(cider_references, cider_predictions_beam)
            cider_beam.append(beam_scores[1]) # index 0 corresponds to dummy data, so we take index 1

            # compute bleu
            beam_bleu = bleu_metric.compute(predictions=predictions_beam, references=references)
            bleu_1_beam.append(beam_bleu['precisions'][0])
            bleu_2_beam.append(beam_bleu['precisions'][1])

            # compute rouge
            beam_rouge = rouge_metric.compute(predictions=predictions_beam, references=references)
            rouge1_beam.append(beam_rouge['rouge1'])
            rougeL_beam.append(beam_rouge['rougeL'])

            # compute meteor
            beam_meteor = meteor_metric.compute(predictions=predictions_beam, references=references)
            meteor_beam.append(beam_meteor['meteor'])

    return bleu_1_greedy, bleu_2_greedy, rouge1_greedy, rougeL_greedy, meteor_greedy, cider_greedy, bleu_1_beam, bleu_2_beam, rouge1_beam, rougeL_beam, meteor_beam, cider_beam

# plot the metric scores
def plot_scores(bleu_1_scores, bleu_2_scores, rouge1_scores, rougeL_scores, meteor_scores, cider_scores, title):
    plt.figure(figsize=(10, 5))
    plt.plot(bleu_1_scores, label='BLEU-1')
    plt.plot(bleu_2_scores, label='BLEU-2')
    plt.plot(rouge1_scores, label='ROUGE-1')
    plt.plot(rougeL_scores, label='ROUGE-L')
    plt.plot(meteor_scores, label='METEOR')
    plt.plot(cider_scores, label='CIDEr')
    plt.xlabel('Epochs')
    plt.ylabel('Scores')
    plt.title(title)
    plt.legend()
    plt.show()

# plot the loss curve
def plot_loss(train_losses, validation_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()
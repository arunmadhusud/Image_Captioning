'''
CS5100 Foundations of Artificial Intelligence
Project
Author: Arun Madhusudhanan, Tejaswini Dilip Deore

- This script is used to write the main code for the project. 
- The main code is used to train the model, validate the model and test the model. 
- The main code also computes the scores and plots the loss and scores.

'''

# Importing the required libraries
import torch
import torchvision.transforms as T
import random
import pandas as pd
from create_dataset import FlickrDataset, collate_fn
from model import ResNet
import os
from tqdm import trange
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from model import CaptioningModel
from utils import train_cnn_lstm_model, validate_cnn_lstm_model, test_model, compute_scores , plot_loss, plot_scores, get_or_build_tokenizer


if __name__ == "__main__":
    
    # Set the seed
    random.seed(42)

    image_data_location = '/home/tejaswini/Projects/Image_Captioning/Images' # Path to the images folder
    caption_data_location = '/home/tejaswini/Projects/Image_Captioning/captions.txt' # Path to the captions file

    df = pd.read_csv(caption_data_location) # Load the captions file

    # Load the resnet features from the file    
    if not os.path.exists('/home/tejaswini/Projects/Image_Captioning/resnet_features.npy'):
        ResNetwork = ResNet(512)
        ResNetwork.eval()
        transforms = T.Compose([T.Resize((224, 224)),T.ToTensor()])
        resnet_features = {}
        with torch.no_grad():
            for i in trange(len(df)):
                image_path = image_data_location + '/' + df.iloc[i, 0]
                img = Image.open(image_path).convert('RGB')
                img = transforms(img).unsqueeze(0)
                img_features = ResNetwork(img)
                if df.iloc[i, 0] not in resnet_features:
                    resnet_features[df.iloc[i, 0]] = img_features

        # save resnet features in a numpy file
        np.save('resnet_features.npy', resnet_features)
    else:
        resnet_features = np.load('/home/tejaswini/Projects/Image_Captioning/resnet_features.npy', allow_pickle=True).item()
    
    # convert tensor to numpy array
    for key in resnet_features.keys():
        resnet_features[key] = resnet_features[key].squeeze()
    
    # Load the tokenizer
    tokenizer_path = 'tokenizer.json'
    tokenizer = get_or_build_tokenizer(df, tokenizer_path)

    transforms = T.Compose([T.Resize((224, 224)),T.ToTensor()])   

    dataset = FlickrDataset(df, tokenizer, image_data_location, resnet_features, transform=transforms)

    # create a train dataloader and test dataloader
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    print('Data loaders created successfully')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    embed_size = 512
    hidden_size = 512
    vocab_size = tokenizer.get_vocab_size()
    # print(vocab_size)
    num_layers = 2
    dropout = 0.5
    learning_rate = 3e-4
    num_epochs = 30

    model = CaptioningModel(embed_size, hidden_size, vocab_size, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<PAD>"))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    validation_losses = []
    best_loss = float('inf')
   
    # Compute metric scores 
    avg_bleu_1_scores_greedy = []
    avg_bleu_2_scores_greedy = []
    avg_rouge1_scores_greedy = []
    avg_rougeL_scores_greedy = []
    avg_meteor_scores_greedy = []
    avg_cider_scores_greedy = []

    avg_bleu_1_scores_beam = []
    avg_bleu_2_scores_beam = []
    avg_rouge1_scores_beam = []
    avg_rougeL_scores_beam = []
    avg_meteor_scores_beam = []
    avg_cider_scores_beam = []

    for epoch in trange(num_epochs):
        train_loss = train_cnn_lstm_model(model, criterion, optimizer, train_loader, device, vocab_size)
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
        training_losses.append(train_loss)
        validation_loss = validate_cnn_lstm_model(model, criterion, test_loader, device, vocab_size)
        print(f'Epoch: {epoch+1}, Validation Loss: {validation_loss}')
        validation_losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), 'model.pth')
            print('Model saved successfully')
        
        #Uncomment the following line to compute metric scores for each epoch
        # # Compute metric scores
        # bleu_1_greedy, bleu_2_greedy, rouge1_greedy, rougeL_greedy, meteor_greedy, cider_greedy, bleu_1_beam, bleu_2_beam, rouge1_beam, rougeL_beam, meteor_beam, cider_beam = compute_scores(model, test_loader, device, tokenizer, df)
        # avg_bleu_1_scores_greedy.append(np.mean(bleu_1_greedy))
        # avg_bleu_2_scores_greedy.append(np.mean(bleu_2_greedy))
        # avg_rouge1_scores_greedy.append(np.mean(rouge1_greedy))
        # avg_rougeL_scores_greedy.append(np.mean(rougeL_greedy))
        # avg_meteor_scores_greedy.append(np.mean(meteor_greedy))
        # avg_cider_scores_greedy.append(np.mean(cider_greedy))

        # avg_bleu_1_scores_beam.append(np.mean(bleu_1_beam))
        # avg_bleu_2_scores_beam.append(np.mean(bleu_2_beam))
        # avg_rouge1_scores_beam.append(np.mean(rouge1_beam))
        # avg_rougeL_scores_beam.append(np.mean(rougeL_beam))
        # avg_meteor_scores_beam.append(np.mean(meteor_beam))
        # avg_cider_scores_beam.append(np.mean(cider_beam))

    print('Training completed')



    # # Test the model 
    # model.load_state_dict(torch.load('/home/tejaswini/Projects/Image_Captioning/model_new.pth'))
    # test_model(model, test_dataset, device, tokenizer)


    # bleu_1_greedy, bleu_2_greedy, rouge1_greedy, rougeL_greedy, meteor_greedy, cider_greedy, bleu_1_beam, bleu_2_beam, rouge1_beam, rougeL_beam, meteor_beam, cider_beam = compute_scores(model, test_loader, device, tokenizer, df)

    # # Print the metric scores
    # print(f'Average BLEU-1 Score for Greedy Search: {np.mean(bleu_1_greedy)}')
    # print(f'Average BLEU-2 Score for Greedy Search: {np.mean(bleu_2_greedy)}')
    # print(f'Average ROUGE-1 Score for Greedy Search: {np.mean(rouge1_greedy)}')
    # print(f'Average ROUGE-L Score for Greedy Search: {np.mean(rougeL_greedy)}')
    # print(f'Average METEOR Score for Greedy Search: {np.mean(meteor_greedy)}')
    # print(f'Average CIDEr Score for Greedy Search: {np.mean(cider_greedy)}')

    # print(f'Average BLEU-1 Score for Beam Search: {np.mean(bleu_1_beam)}')
    # print(f'Average BLEU-2 Score for Beam Search: {np.mean(bleu_2_beam)}')
    # print(f'Average ROUGE-1 Score for Beam Search: {np.mean(rouge1_beam)}')
    # print(f'Average ROUGE-L Score for Beam Search: {np.mean(rougeL_beam)}')
    # print(f'Average METEOR Score for Beam Search: {np.mean(meteor_beam)}')
    # print(f'Average CIDEr Score for Beam Search: {np.mean(cider_beam)}')

    # Plot the training and validation losses
    plot_loss(training_losses, validation_losses)

    # # Plot the metric scores
    # plot_scores(avg_bleu_1_scores_greedy, avg_bleu_2_scores_greedy, avg_rouge1_scores_greedy, avg_rougeL_scores_greedy, avg_meteor_scores_greedy, avg_cider_scores_greedy, title='Greedy Search')
    # plot_scores(avg_bleu_1_scores_beam, avg_bleu_2_scores_beam, avg_rouge1_scores_beam, avg_rougeL_scores_beam, avg_meteor_scores_beam, avg_cider_scores_beam, title='Beam Search')



    






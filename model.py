'''
CS5100 Foundations of Artificial Intelligence
Project
Author: Arun Madhusudhanana, Tejaswini Dilip Deore

This script is used to create the model for the image captioning task. 
'''
# Importing the required libraries
import torchvision.models as models
import torchvision.transforms as T
import torch
import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    '''
    Class to create the ResNet50 model.
    '''
    def __init__(self, embed_size):
        super(ResNet, self).__init__()
        # Load the pretrained ResNet-50 model
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False) 
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)    
        return features


class EncoderCNN(nn.Module):
    '''
    Class to create the EncoderCNN model.
    '''
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.embed = nn.Linear(2048, embed_size)

    def forward(self, image_features):
        features = self.embed(image_features)
        return features

class DecoderRNN(nn.Module):
    '''
    Class to create the DecoderRNN model. 
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1,dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        # print(embeddings.shape)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        # print(lstm_out.shape)
        outputs = self.linear(lstm_out)
        return outputs
    
    def sample_caption(self, inputs, hidden = None, max_len=20, tokenizer = None):
        '''
        This function is used to sample the caption from the model using Greedy Search.
        Args:
            inputs (tensor): Tensor containing the input features
            hidden (tensor): Tensor containing the hidden state
            max_len (int): Maximum length of the caption
            tokenizer (Tokenizer): Tokenizer object to decode the caption
        Returns:
            output (str): The sampled caption
        '''
        output = []
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.linear(lstm_out)
            outputs = outputs.view(inputs.size(0), -1)

            predicted = outputs.argmax(1) # greedy search
            output.append(predicted.item())
            # print(predicted)

            if predicted.item() == tokenizer.token_to_id("<EOS>"): 
                break
                
            inputs = self.embed(predicted.unsqueeze(0))

        return tokenizer.decode(output)
    
    def beam_search(self, features, hidden = None, max_len=20, tokenizer = None, beam_size=1):
        '''
        This function is used to sample the caption from the model using Beam Search.
        Args:
            features (tensor): Tensor containing the input features
            hidden (tensor): Tensor containing the hidden state
            max_len (int): Maximum length of the caption
            tokenizer (Tokenizer): Tokenizer object to decode the caption
            beam_size (int): Beam size for the search
        Returns:
            output (str): The sampled caption
        '''
        output = []
        start_token = tokenizer.token_to_id("<SOS>") # start token
        features = features.unsqueeze(1)
        candidates = [(torch.tensor([start_token]), 0)] # create a list of candidates with the start token
        for t in range(max_len-1):
            next_candidates = []
            for c in range(len(candidates)):
                # print(candidates[c][0].device)
                candidates_embed = self.embed(candidates[c][0].to(features.device))
                # print(candidates_embed.device)
                inputs = torch.cat((features, candidates_embed.unsqueeze(0)), 1)
                lstm_out, hidden = self.lstm(inputs, hidden)
                outputs = self.linear(lstm_out[:, -1, :])
                outputs_softmax = nn.functional.log_softmax(outputs, dim=1)        
                _,words = outputs.topk(beam_size)
                probs,_ = outputs_softmax.topk(beam_size)          
                for i in range(beam_size):                    
                    new_candidate = torch.cat((candidates[c][0].to(features.device) , words[:, i]))
                    new_score = candidates[c][1] - probs[:, i].item()
                    next_candidates.append((new_candidate, new_score))
            next_candidates = sorted(next_candidates, key=lambda x: x[1]) # sort the candidates based on the score 
            candidates = next_candidates[:beam_size] # select the top beam_size candidates

            if candidates[0][0][-1] == tokenizer.token_to_id("<EOS>"):
                break

        output = tokenizer.decode(candidates[0][0].tolist())
        return output
    
    
class CaptioningModel(nn.Module):
    '''
    Class to create the CaptioningModel model.
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super(CaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, dropout)
    
    def forward(self, image_features, captions):
        features = self.encoder(image_features)
        outputs = self.decoder(features, captions)
        return outputs
    
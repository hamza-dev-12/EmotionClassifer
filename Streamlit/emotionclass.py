import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

class Emotion_Sentence_Classifier(nn.Module):

  def __init__(self,config : dict):
    super().__init__()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'],return_dict=True)
    self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size,self.pretrained_model.config.hidden_size)
    self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size,self.config['n_labels'])

    #initilizing weights
    torch.nn.init.xavier_uniform(self.classifier.weight)

    #loss function
    self.loss_func = nn.BCEWithLogitsLoss(reduction = 'mean')
    #BCE is the combination of cross entropy followed by the sigmoid function

    #droupout for model to avoid to be overfitted
    self.dropout = nn.Dropout()

    # for param in self.pretrained_model.parameters():
    #         param.requires_grad = False

  def forward(self,input_ids,attention_mask,labels=None):
  #roberta layer
    output = self.pretrained_model(input_ids = input_ids,attention_mask = attention_mask)
    #print('output before avg-pooling: ',output)
    # batch size * sequence_length * hidden layer size
    pooled_output = torch.mean(output.last_hidden_state,1)
    #batch size * hidden layer size
    #print('after avg pooling : ',pooled_output)
    #my nn layer
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = F.relu(pooled_output) # relu activation function
    logits = self.classifier(pooled_output)
    loss = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1,self.config['n_labels']),labels.view(-1,self.config['n_labels']))
    return loss , logits


from distutils.command.config import config
from content_model_pytorch.config import ModelConfig
import torch
import torch.nn as nn
import math

mconfig = ModelConfig()

class ContentModel(nn.Module):
    """
    Model architecture to obtain content embedding
    """

    def __init__(self):
        super(ContentModel, self).__init__()
        self.input_size = mconfig.input_size
        self.layers = []
        input_size = self.input_size
        self.layers.append(nn.Linear(input_size,mconfig.layers_size))
        for i in range(mconfig.n_hidden_layers):
            self.layers.append(nn.Linear(mconfig.layers_size, mconfig.layers_size))
        self.layers.append(nn.Linear(mconfig.layers_size, mconfig.embedding_size))
        self.content_classifier = nn.Linear(mconfig.embedding_size, mconfig.vocab_size)
        self.dropout = nn.Dropout(mconfig.dropout)
    
    def forward(self, sentences, content_bow):
        '''
            Args:
            sentences => embeddings of sentences, shape: (batch_size*input_size)
            content_bow => bag of words of input sentences, shape: (batch_size*bow_size)
        '''
        '''
            Output:
            class_loss => loss incurred by content classification from content embedding
        '''     

        input = sentences
        for layer in self.layers:
            output = layer(input)
            input = output
        content_embedding = output

        return self.get_content_class_loss(content_embedding,content_bow)

    def get_content_class_loss(self, content_embedding, content_bow):
        """
            this loss calculate the ammount of content information actually
            preserved in content embedding
            Return:
            cross entropy loss of content classifier
        """
        # predictions
        preds = nn.Softmax(dim=1)(self.dropout(content_embedding))
        # check smoothing
        # BCELoss
        return nn.BCELoss()(preds, content_bow)
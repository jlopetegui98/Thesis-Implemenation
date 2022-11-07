from distutils.command.config import config
from content_model_pytorch.config import ModelConfig
import torch
import torch.nn as nn
import math


class ContentModel(nn.Module):
    """
    Model architecture to obtain content embedding
    """

    def __init__(self, config):
        super(ContentModel, self).__init__()
        self.mconfig = config
        self.input_size = self.mconfig.input_size
        self.layers = []
        input_size = self.input_size
        self.layers.append(nn.Linear(input_size, self.mconfig.layers_size))
        for i in range(self.mconfig.n_hidden_layers):
            self.layers.append(
                nn.Linear(self.mconfig.layers_size, self.mconfig.layers_size)
            )
        self.layers.append(
            nn.Linear(self.mconfig.layers_size, self.mconfig.embedding_size)
        )
        self.content_classifier = nn.Linear(
            self.mconfig.embedding_size, self.mconfig.vocab_size
        )
        self.dropout = nn.Dropout(self.mconfig.dropout)

    def forward(self, sentences, content_bow):
        """
        Args:
        sentences => embeddings of sentences, shape: (batch_size*input_size)
        content_bow => bag of words of input sentences, shape: (batch_size*bow_size)
        """
        """
            Output:
            class_loss => loss incurred by content classification from content embedding
        """

        input = sentences
        for layer in self.layers:
            output = layer(input)
            input = output
        content_embedding = output

        return self.get_content_class_loss(content_embedding, content_bow)

    def get_content_class_loss(self, content_embedding, content_bow):
        """
        this loss calculate the ammount of content information actually
        preserved in content embedding
        Return:
        cross entropy loss of content classifier
        """
        # predictions
        output = self.content_classifier(self.dropout(content_embedding))
        preds = nn.Softmax(dim=0)(output)
        # check smoothing
        smoothed_content_bow = (
            content_bow * (1 - self.mconfig.label_smoothing)
            + self.mconfig.label_smoothing / self.mconfig.vocab_size
        )

        # BCELoss
        return nn.BCELoss()(preds, smoothed_content_bow)

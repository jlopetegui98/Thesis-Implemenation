import torch.nn as nn
import torch


class StyleModel(nn.Module):
    """
    Model to obtain style embedding of a given text
    using as inputs text embedding and content embedding
    """

    def __init__(self, config):
        super(StyleModel, self).__init__()

        self.config = config
        input_size = config.embedding_size
        self.encoder = nn.Linear(input_size, config.style_embedding_size)
        self.decoder = nn.Linear(
            config.style_embedding_size + config.content_embedding_size, input_size
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, text_embeddings, content_embeddings):
        """
        Args:
        text_embeddings => embeddings of sentences, shape: (batch_size*input_size)
        content_embeddings => content embeddings of input sentences, shape: (batch_size*content_embedding_size)
        """
        """
            Output:
            style_embedding => style embedding of input sentences, shape: (batch_size*style_embedding_size)
            decoded_embedding => decoded embedding from style and content embedding, shape: (batch_size*input_size)
        """
        style_embedding = self.encoder(text_embeddings)
        style_embedding = self.dropout(style_embedding)
        # concat style and content embedding
        decoded_embedding = torch.cat((style_embedding, content_embeddings), 1)
        decoded_embedding = self.decoder(decoded_embedding)
        return style_embedding, decoded_embedding

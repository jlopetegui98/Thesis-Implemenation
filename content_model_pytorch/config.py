class ModelConfig():
    """
        model configuration
    """
    def __init__(self, embedding_size = 10, number_of_layers = 3) -> None:
        self.vocab_size = 30522
        self.input_size = 768
        self.layers_size = 768//2
        self.epoch = 20
        # content embedding size
        self.embedding_size = embedding_size
        # number of Linear layers
        self.n_hidden_layers = number_of_layers
        # batch
        self.batch_size = 128
        # dropout
        self.dropout = 0.8
        # learning rate
        self.lr = 0.001
        

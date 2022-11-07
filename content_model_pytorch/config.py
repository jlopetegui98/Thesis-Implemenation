class ModelConfig:
    """
    model configuration
    """

    def __init__(self, vocab_size, embedding_size=10, number_of_layers=3) -> None:
        self.vocab_size = vocab_size
        self.input_size = 768
        self.layers_size = 768 // 2
        self.epoch = 10
        # content embedding size
        self.embedding_size = embedding_size
        # number of Linear layers
        self.n_hidden_layers = number_of_layers
        # batch
        self.batch_size = 50
        # dropout
        self.dropout = 0.5
        # learning rate
        self.lr = 0.001
        self.label_smoothing = 0.1

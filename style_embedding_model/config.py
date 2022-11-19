class ModelConfig:
    """
    model configuration
    """

    def __init__(
        self,
        embedding_size=10,
        style_embedding_size=50,
        content_embedding_size=100,
        epochs=10,
    ) -> None:
        self.epochs = epochs
        # text embedding size
        self.embedding_size = embedding_size
        # content embedding size
        self.content_embedding_size = content_embedding_size
        # style embedding size
        self.style_embedding_size = style_embedding_size
        # batch
        self.batch_size = 100
        # dropout
        self.dropout = 0.2
        # learning rate
        self.lr = 0.001
        self.label_smoothing = 0.1

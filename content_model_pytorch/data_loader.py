from .bow_model.bow_model import Vocab
from torch.utils.data import Dataset
import torch

# class for dataset of sentences used to train content model
class SentencesDataset(Dataset):
    def __init__(
        self,
        sentences_embeddings_path,
        content_words_sentences,
        mode="train",
        word2indx=None,
        batches_size=100,
    ):
        self.sentences_embeddings_path = sentences_embeddings_path
        self.content_words_sentencs = content_words_sentences
        self.mode = mode
        self.word2indx = word2indx
        self.vocab = self._get_vocab()
        self.batch_size = batches_size

    # create Vocab instance from content words sentences
    def _get_vocab(self):
        if self.mode == "train":
            return Vocab(self.content_words_sentences)
        else:
            return Vocab(None, self.word2indx)

    # get item at index, return sentence embedding and bow representation of sentence
    def __getitem__(self, index):
        """
        return sentence_embedding
        return bow representation of sentence's content
        """
        bath_idx = index // self.batch_size
        sent_idx = index % self.batch_size
        sentence_embedding = torch.load(
            f"{self.sentences_embeddings_path}{self.mode}_batch_{bath_idx+1}.pt"
        )[sent_idx]
        return sentence_embedding, self.vocab.get_bow_representation(
            self.content_words_sentences[index]
        )

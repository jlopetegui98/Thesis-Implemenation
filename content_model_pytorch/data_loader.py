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
        self.content_words_sentences = content_words_sentences
        self.mode = mode
        self.word2indx = word2indx
        self.vocab = self._get_vocab()
        self.batch_size = batches_size

    # create Vocab instance from content words sentences
    def _get_vocab(self):
        if self.mode == "train":
            return Vocab([sentence for sentence, _ in self.content_words_sentences])
        else:
            return Vocab(None, word2index=self.word2indx)

    # get item at index, return sentence embedding and bow representation of sentence
    def __getitem__(self, index):
        """
        return sentence_embedding
        return bow representation of sentence's content
        """
        # fix_index is index without removing sentences without content words
        fix_index = self.content_words_sentences[index][1]

        batch_idx = fix_index // self.batch_size
        sent_idx = fix_index % self.batch_size
        sentence_embedding = torch.load(
            f"{self.sentences_embeddings_path}{self.mode}_batch_{batch_idx+1}.pt"
        )[sent_idx]
        return sentence_embedding, self.vocab.get_bow_representation(
            self.content_words_sentences[index][0]
        )

    # get length of dataset
    def __len__(self):
        return len(self.content_words_sentences)

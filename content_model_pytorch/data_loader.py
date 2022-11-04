from bow_model.bow_model import Vocab
from torch.utils.data import Dataset

# class for dataset of sentences used to train content model
class SentencesDataset(Dataset):
    def __init__(
        self,
        sentences_embeddings,
        content_words_sentences,
        mode="train",
        word2indx=None,
    ):
        self.sentences = sentences_embeddings
        self.content_words_sentences = content_words_sentences
        self.mode = mode
        self.word2indx = word2indx
        self.vocab = self._get_vocab()

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

        return self.sentences[index], self.vocab.get_bow_representation(
            self.content_words_sentences[index]
        )

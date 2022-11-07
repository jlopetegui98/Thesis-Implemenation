import collections
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
import torch
import numpy as np


class Vocab:
    def __init__(
        self, sentences_content=None, percent_reduction=0.8, word2index={}
    ) -> None:
        self.sentences = sentences_content
        self.vocab_size_reduction = percent_reduction
        self.word2indx = word2index
        self.indx2word = {}
        self.vocab = self._read_vocab()

    def _read_vocab(self):
        # if sentences are provided, create vocab from sentences
        vocab = collections.Counter()
        if self.sentences:
            for sentence in self.sentences:
                for word in sentence.split():
                    vocab[word] += 1
            vocab = self._filter_vocab(vocab)
            vocab = vocab.most_common(int(len(vocab) * self.vocab_size_reduction))
            vocab = {word: index for index, (word, count) in enumerate(vocab)}
            self.word2indx = vocab
            self.indx2word = {index: word for word, index in vocab.items()}
            return vocab
        # if sentences are not provided, use word2index
        else:
            self.indx2word = {index: word for word, index in self.word2indx.items()}
            return vocab.update(self.word2indx.keys())
        # # read vocab from file
        # if self.sentences is None:
        #     vocab = collections.Counter()
        #     for word, index in self.word2indx.items():
        #         self.indx2word[index] = word

        # vocab = collections.Counter()
        # index = 1
        # with open(self.config.vocab_path, "r") as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         words = line.split()
        #         for word in words:
        #             if word not in self.word2indx:
        #                 self.word2indx[word] = index
        #                 self.indx2word[index] = word
        #                 index += 1
        #         vocab.update(words)
        # vocab = self._filter_vocab(vocab)
        # return vocab.most_common(int(self.vocab_size_reduction * len(vocab)))

    # remove stop words from vocab
    def _filter_vocab(self, vocab) -> dict:
        stop_words = set(stopwords.words("english"))
        stop_words.update(spacy_stopwords)

        for word in stop_words:
            if word in vocab:
                del vocab[word]
        return vocab

    # get bow representation of a sentence
    def get_bow_representation(self, sentence) -> torch:
        bow = np.zeros(len(self.word2indx), dtype=np.float32)
        for word in sentence:
            if word in self.word2indx:
                bow[self.word2indx[word]] += 1
        bow /= max(sum(bow), 1)
        return torch.from_numpy(bow)

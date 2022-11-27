from torch.utils.data import Dataset
import torch

# class for dataset of sentences used to train style model
class SentencesCDataset(Dataset):
    def __init__(
        self,
        sentences_embeddings_path,
        content_embedding_model,
        content_words_sentences,
        batches_size=100,
    ):
        self.sentences_embeddings_path = sentences_embeddings_path
        self.content_embedding_model = content_embedding_model
        self.content_embedding_model.eval()
        self.content_words_sentences = content_words_sentences
        self.batch_size = batches_size

    # get item at index, return sentence embedding and bow representation of sentence
    def __getitem__(self, index):
        """
        return sentence_embedding
        return content_embedding
        """
        # fix_index is index without removing sentences without content words
        fix_index = self.content_words_sentences[index][1]

        batch_idx = fix_index // self.batch_size
        sent_idx = fix_index % self.batch_size
        sentence_embedding = torch.load(
            f"{self.sentences_embeddings_path}{self.mode}_batch_{batch_idx+1}.pt"
        )[sent_idx]
        content_embedding, _ = self.content_embedding_model(
            sentence_embedding, mode="eval"
        )
        return (sentence_embedding, content_embedding)

    # get length of dataset
    def __len__(self):
        return len(self.content_words_sentences)

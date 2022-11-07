from torch.utils.data import Dataset


class SentDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = [tokenizer(sentence) for sentence in sentences]

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)

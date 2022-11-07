from torch.utils.data import DataLoader
import torch
from transformers import (
    BertTokenizer,
    BertModel,
    BertTokenizer,
    DataCollatorWithPadding,
)
from embedding_model.data_loader import SentDataset
from tqdm import tqdm


class Embedder:
    def __init__(self, batch_size=100):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=True
        )
        self.model.eval()
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.batch_size = batch_size

    def get_embeddings(self, sentences, device, path_to_save):
        dataset = SentDataset(sentences, self.tokenizer)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=self.data_collator
        )
        self.model.to(device)
        i = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch.to(device)
                out = self.model(**batch)
                i += 1
                torch.save(out.pooler_output, path_to_save + f"batch_{i}.pt")
                del batch
                del out  # esto fue lo q t dije q m falto probar
                torch.cuda.empty_cache()

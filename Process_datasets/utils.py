import time
import torch
from transformers import BertModel
import logging

def get_embeddings(sentences,embeddings,tokenizer,i,j):
    logging.info("Getting embeddings for sentences {} to {}".format(i,j))
    for k in range(i,min(j,len(sentences))):
        sentence = sentences[k]
        marked_sentence = "[CLS] " + sentence + " [SEP]"
        tokens = tokenizer.tokenize(marked_sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_tensor = torch.tensor([token_ids])
        model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        model.eval()
        with torch.no_grad():
            outputs = model(token_tensor)
            hidden_states = outputs[2]
        
        token_vecs = hidden_states[-2][0]

        sentence_embedding = torch.mean(token_vecs, dim=0)

        embeddings[k] = sentence_embedding
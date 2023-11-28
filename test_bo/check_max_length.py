from datasets import load_dataset
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

dataset_path = ["Dahoas/rm-static"]
dset = load_dataset(*dataset_path, split="train")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
similarity_model = "all-mpnet-base-v1"
embedder = SentenceTransformer(similarity_model).to(device)

max_length = 0
for i in tqdm(range(len(dset))):
    text = dset[i]["prompt"] + dset[i]["response"]
    # tokenized_text = embedder.tokenize(text)
    # current_length = len(tokenized_text["input_ids"])
    current_length = len(text.split(" "))
    max_length = max(max_length, current_length)
    
    if (i % 1000 == 0):
        print("Current max length: ", max_length)
    
print("Max length: ", max_length)
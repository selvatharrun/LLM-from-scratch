import torch
from torch.utils.data import DataLoader,Dataset
import tiktoken

class GPTdatasetv1(Dataset):

    def __init__(self,text,tokenizer,maxlen,stride):
        self.input =[]
        self.target =[]

        tokenizer = tokenizer.encode(text, allowed_special = "<|endoftext|>")
        
        for i in range(0,len(tokenizer),stride):
            self.input.append(torch.tensor(tokenizer[i:i+maxlen]))
            self.target.append(torch.tensor(tokenizer[i+1:i+maxlen+1]))
        
    def __len__(self):
        return len(self.input)
    
    def __getid__(self,idx):
        return self.input[idx],self.target[idx]
    
def createdataloader(txt,batchsize=4,maxlen=256,stride=128,shuffle = True,drop_last = True,num_workers=1):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTdatasetv1(txt,tokenizer,maxlen,stride)

    dataloader = DataLoader(
                    dataset,
                    batch_size = batchsize,
                    shuffle = shuffle,
                    drop_last=drop_last,
                    num_workers = num_workers
    )

    return dataloader


"""
text = "In the field of machine learning, subword tokenization is important.
It helps to handle unknown or rare words by breaking them down into known chunks.
This is especially useful in generative language models."

dataloader, tokenizer = create_dataloader_v1(text, batch_size=2, max_length=16, stride=8)

for batch in dataloader:
    input_ids, target_ids = batch
    print("Input IDs:", input_ids)
    print("Decoded Input:", [tokenizer.decode(x.tolist()) for x in input_ids])
    print("Target IDs:", target_ids)
    print("Decoded Target:", [tokenizer.decode(y.tolist()) for y in target_ids])
    break
"""


from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
import pandas
from pandas import DataFrame
import torch

class CustomDataset(Dataset):
    """
    CustomDataset for this project. It is a wrapper for the data to access it via PyTorch Dataloader
    
    based on: mona/ml/custom_dataset
    Changes:
    - truncation method was changed because the previous one caused this error:
        "RuntimeError: stack expects each tensor to be equal size, but got [206] at entry 0 and [295] at entry 4"
    """
    
    
    def __init__(self, dataframe, tokenizer, max_len):
        """
        params:
            dataframe : pandas dataframe that contains all data.
            tokenizer : Huggingface tokenizer used for tokenizing every textual data in the dataframe.
            max_len : maximum length of tokens per sample
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.input_text = list(dataframe.text)
        self.targets = list(self.data.labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])
        input_text = " ".join(input_text.split())
        
        inputs = self.tokenizer.encode_plus(
            text=input_text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation= 'longest_first',
            #truncation= 'only_second',
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
    def from_csv(tag, tokenizer, max_len, korpus_path, text_keyword):
        """
        Creates three (train,test,validate) CustomDatasets from one csv-file.
        """
        
        korpus_df = pandas.read_csv(korpus_path, dtype=str, delimiter='|')
        df = DataFrame()
        df['text'] = korpus_df[text_keyword]
        df['dataset'] = korpus_df['dataset']
        # convert the label string to list of ints
        df['labels'] = korpus_df[tag].apply(lambda labels_string: [int(label_char) for label_char in labels_string])
        
        train = df[df.dataset == 'train']
        test = df[df.dataset == 'test']
        validate = df[df.dataset == 'validate']
        
        return CustomDataset(train, tokenizer, max_len), CustomDataset(test, tokenizer, max_len), CustomDataset(validate, tokenizer, max_len)
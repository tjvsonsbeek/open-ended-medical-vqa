
from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from transformers import set_seed, GPT2Config, GPT2Tokenizer
from transformers import AutoTokenizer
from transformers.models.biogpt import BioGptTokenizer
import os
import pandas as pd
from torch.utils.data import Dataset
import pickle
from torch.utils.data import DataLoader, random_split
import numpy as np
import pdb

class medvqaDataset(Dataset):
    def __init__(self, path, split='train',like_test=False,prefix_length=2,model_type = 'gpt2',setting='oe', abl='none'):
        super().__init__()
        data_path = path+split+'.pkl'
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        # print("Data size is %0d" % len(all_data["txt_prefix"]))
        sys.stdout.flush()
        self.model_type = model_type

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.abl=abl
        self.img_ids = all_data["img_ids"]
        # self.txt_prefixes = all_data["txt_prefix"]
        self.img_prefixes = all_data["img_prefix"]
        self.questions_raw = all_data['questions']
        self.answers_raw = all_data['answers']

        self.max_seqs_len = all_data['max_seqs_len']
        self.classes = all_data['class_ids']       

        self.train_setting = True if (split!='test'and like_test==False) else False
        self.prefix_len = prefix_length
        self.setting = setting
        
    def __len__(self):
        return len(self.answers_raw)
    def pad_sequences(self,index):
        m = [torch.tensor(self.tokenizer.encode('question: ')),torch.tensor(self.tokenizer.encode(' context:')),torch.tensor(self.tokenizer.encode('answer ')),torch.tensor(self.tokenizer.encode('<|endoftext|>'))]
        m_mask = [torch.ones(len(self.tokenizer.encode('question: '))),torch.ones(len(self.tokenizer.encode(' context:'))),torch.ones(len(self.tokenizer.encode('answer '))),torch.zeros(self.tokenizer.encode('<|endoftext|>'))]   

        if self.train_setting:
            # construct the model input. The order is question, image, answer. During training the answer is masked. Any padding is placed on the right of the sequence. 
            # placeholder tokens are used on the location where the visual prefix will be inserted, with q_len indicating this location. 
            q=torch.tensor(self.tokenizer.encode(self.questions_raw[index]))
            a=torch.tensor(self.tokenizer.encode(str(self.answers_raw[index])))
            
            q,q_mask,leftover_tokens = self.make_padding(self.max_seqs_len[0],q,question=True)
            if self.abl=='remove_question': leftover_tokens=0
            a,a_mask,_ = self.make_padding(self.max_seqs_len[1],a,leftover_tokens=leftover_tokens)
            if len((a==0).nonzero())!=0:
                pad_start = (a==0).nonzero()[0]
            else:
                pad_start=[]
            a = torch.cat((a,m[3])) if len(pad_start)==0 else torch.cat((a[:pad_start],m[3],a[pad_start:]))
            
            if self.abl=='replace_visual':
                q_len2 = m[0].size(0) + q.size(0) + m[1].size(0)+self.prefix_len
                q_len = m[0].size(0) + q.size(0) + m[1].size(0)
                q = torch.cat((m[0],q,m[1],torch.ones(self.prefix_len),m[2],a))
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.ones(self.prefix_len),m_mask[2],a_mask,m_mask[3]))
            elif self.abl=='remove_visual':
                q_len2 = m[0].size(0) + q.size(0) 
                q_len = m[0].size(0) + q.size(0) + m[1].size(0)
                q = torch.cat((m[0],q,m[2],a))
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[2],a_mask,m_mask[3]))
            elif self.abl=='remove_question':
                q_len2 = m[1].size(0)+self.prefix_len
                q_len = m[1].size(0)
                q = torch.cat((m[1],torch.ones(self.prefix_len),m[2],a))
                q_mask = torch.cat((m_mask[1],torch.ones(self.prefix_len),m_mask[2],a_mask,m_mask[3]))
                
            elif self.abl=='swap':
                q_len2 = m[0].size(0) + q.size(0) + m[1].size(0)+self.prefix_len
                q_len = m[1].size(0) 
                q = torch.cat((m[1],torch.ones(self.prefix_len),m[0],q,m[2],a))
                q_mask = torch.cat((m_mask[1],torch.ones(self.prefix_len),m_mask[0],q_mask,m_mask[2],a_mask,m_mask[3]))
                
            return q,q_mask, q_len,q_len2
        else:
            q = torch.tensor(self.tokenizer.encode(self.questions_raw[index]))
            q,q_mask,_ = self.make_padding_only_question(self.max_seqs_len[0],q)
            q_len2 = m[0].size(0) + q.size(0) + m[1].size(0)
            if self.abl=='replace_visual':
                q_len2 = m[0].size(0) + q.size(0) + m[1].size(0)+self.prefix_len                
                q_len = m[0].size(0) + q.size(0) + m[1].size(0)
                q = torch.cat((m[0],q,m[1],torch.ones(self.prefix_len),m[2]))
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.ones(self.prefix_len),m_mask[2]))
                
            elif self.abl=='remove_visual':
                q_len2 = m[0].size(0) + q.size(0) 
                q_len = m[0].size(0) + q.size(0) 
                q = torch.cat((m[0],q,m[2]))
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[2]))
                
            elif self.abl=='remove_question':
                q_len2 = m[1].size(0)+self.prefix_len
                q_len = m[1].size(0)
                q = torch.cat((m[1],torch.ones(self.prefix_len),m[2]))
                q_mask = torch.cat((m_mask[1],torch.ones(self.prefix_len),m_mask[2]))
                
            elif self.abl=='swap':
                q_len2 = m[0].size(0) + q.size(0) + m[1].size(0)+self.prefix_len
                q_len = m[1].size(0) 
                q = torch.cat((m[1],torch.ones(self.prefix_len),m[0],q,m[2]))
                q_mask = torch.cat((m_mask[1],torch.ones(self.prefix_len),m_mask[0],q_mask,m_mask[2]))
                
            return q,q_mask,q_len,q_len2


        
            
    def make_padding(self, max_len, tokens, question=False,leftover_tokens=0):
        padding = max_len - tokens.size(0) 
        if padding > 0:
            if question:
                leftover_tokens = padding
                mask = torch.ones(tokens.size(0))
            else:
                tokens = torch.cat((tokens, torch.zeros(padding+leftover_tokens)))
                mask = torch.zeros(max_len+leftover_tokens)    
              
        elif padding==0:
            if question:
                mask = torch.ones(tokens.size(0)) 
            else:
                mask = torch.zeros(tokens.size(0)+leftover_tokens)
                tokens = torch.cat((tokens,torch.zeros(leftover_tokens)))
        elif padding < 0:
            if question:
                tokens = tokens[:max_len]
                mask = torch.ones(max_len)
            else:
                tokens = torch.cat((tokens[:max_len], torch.zeros(leftover_tokens)))
                mask = torch.zeros(max_len+ leftover_tokens)
        return tokens, mask, leftover_tokens
    def make_padding_only_question(self, max_len, tokens,do_padding=False):
        padding = max_len - tokens.size(0)
        padding_len = 0
        if padding > 0:
            if do_padding:
                mask = torch.cat((torch.ones(tokens.size(0)),torch.zeros(padding)))
                tokens = torch.cat((tokens,torch.zeros(padding)))
                padding_len = padding
            else:
                mask = torch.ones(tokens.size(0))
        elif padding ==0:
            mask = torch.ones(max_len)
        elif padding < 0:
            tokens = tokens[:max_len]
            mask = torch.ones(max_len)
        return tokens, mask, padding_len
            
    def __getitem__(self, index):
        prefix = self.img_prefixes[self.img_ids[index]]
        tokens, mask, q_len, q_len2  = self.pad_sequences(index)
        return prefix,  self.classes[index], tokens, mask, q_len, q_len2

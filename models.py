import numpy as np
from tqdm import tqdm
import sys
import os
import pdb
from typing import Tuple, Optional, Union

from peft import LoraConfig, get_peft_model,get_peft_config,PeftModelForCausalLM,TaskType,PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig 

import torch
import torch.nn as nn
from torch.nn import functional as nnf

import transformers
from transformers import set_seed, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.biogpt import BioGptForCausalLM, BioGptTokenizer, BioGptConfig
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig

from prefix_mappers import MLP, TransformerMapper

class VQAmedModel(nn.Module):
    def forward(self, prefix, labels, tokens, mask, q_len, batch_size):
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        if self.gpttype=='microsoft/biogpt':
            embedding = self.gpt.transformer.embed_tokens(tokens)
        else:
            embedding = self.gpt.transformer.wte(tokens)
        for b in range(batch_size):
            # insert the visual prefix after the question 
            embedding[b,q_len[b]:q_len[b]+self.prefix_length,:] = prefix_projections[b]  
        return self.gpt(inputs_embeds=embedding, attention_mask=mask)
    def generate(self, prefix, labels, tokens, mask, q_len):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
        if self.gpttype=='microsoft/biogpt':
            embedding_txt = self.gpt.transformer.embed_tokens(tokens)
        else:
            embedding_txt = self.gpt.transformer.wte(tokens)
        embedding_txt[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embedding_txt
    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="lora",
        mapping_type="MLP",
        args=None,
    ):
        super(VQAmedModel, self).__init__()
        gpttype = args.model_type
        self.gpttype = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        self.gpt = AutoModelForCausalLM.from_pretrained(gpttype,load_in_8bit=True,device_map='auto')
        # load the relevant fine-tuning strategy 
        if setting == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == "MLP":
            self.clip_project = MLP((
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")


# adaptation of VQAmedModel for ablation studies
class VQAmedModel_abl(nn.Module):
    def forward(self, prefix, labels, tokens, mask, q_len, batch_size,abl):
        embeddings = self.gpt.transformer.wte(tokens)
        if abl=="replace_visual":
            for b in range(batch_size):
                embeddings[b,q_len[b]:q_len[b]+self.prefix_length,:] = self.nv_tokens[b]  
        elif abl=="remove_question":
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
        elif abl=="swap":
            prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
            embeddings[:,q_len[0]:q_len[0]+self.prefix_length,:] = prefix_projections
        return self.gpt(inputs_embeds=embeddings, attention_mask=mask)

    def generate(self, prefix, labels, tokens, mask, q_len,abl):
        prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
        embeddings = self.gpt.transformer.wte(tokens)
        if abl=="replace_visual":
            embeddings[q_len:q_len+self.prefix_length,:] = self.nv_tokens[0]  
        elif abl=="remove_question":
            prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
            embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
        elif abl=="swap":
            prefix_projections = self.clip_project(prefix.view(1, -1)).view(self.prefix_length, self.gpt_embedding_size)
            embeddings[q_len:q_len+self.prefix_length,:] = prefix_projections
        return embeddings

    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="frozen",
        mapping_type="MLP",
        args=None,
    ):
        super(VQAmedModel_abl, self).__init__()
        gpttype = "gpt2-xl"
        self.model_type = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpttype,load_in_8bit=True,device_map='auto')
        if setting == "lora":
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prefixtuning":
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="p_tuning":
            peft_config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=="prompttuning":
            peft_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=30)
            self.gpt = get_peft_model(self.gpt,peft_config)
        elif setting=='frozen':
            for param in self.gpt.transformer.parameters():
                param.requires_grad = False
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # for the replace_visual ablation study we replace the visual tokens with learnable parameters 
        self.nv_tokens = torch.nn.Parameter(torch.randn(args.batch_size,prefix_length,self.gpt_embedding_size),requires_grad=True).cuda()
        if mapping_type == "MLP":
            self.clip_project = MLP((prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")
        
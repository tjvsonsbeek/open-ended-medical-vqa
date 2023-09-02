from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score,roc_auc_score
from utils import generate_beam
from nltk.translate.bleu_score import sentence_bleu
from transformers import GPT2Tokenizer
import pdb
from evaluate import load
import collections
from torch.cuda.amp import autocast
import os
    
def eval_gpt_open_ended(model, dataset, args, print_vis_token_meaning=True):
    model.eval()
    model=model.cuda()
    bert_score = load("bertscore")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    bleu_avg1=0.
    bert_avg1 = 0.
    bert_avg2 = 0.
    bert_avg3 = 0.
    f1_avg = 0. 
    acc = 0.
    acc_oe = 0.
    acc_yn = 0.
    c_oe =1e-9
    c_yn =1e-9 
    with tqdm(total=len(dataset)) as epoch_pbar:
        epoch_pbar.set_description("Testing")
        for item in range(len(dataset)):
            prefix,  labels, tokens, mask, q_len = dataset[item]
            prefix = prefix.type(torch.float32).cuda()
            tokens = tokens.type(torch.long).cuda()
            mask = mask.cuda()
            with autocast(dtype=torch.float16):
              with torch.no_grad():
                  embed = model.generate(prefix,labels,tokens,mask,q_len).view(1,tokens.size(0),-1)
                  if print_vis_token_meaning:
                    prefix_projections = embed[:,q_len:q_len+model.prefix_length,:]
                    for i in range(prefix_projections.size(1)):
                      print_nearest_text_token(prefix_projections[0,i], model)
                  out_text = generate_beam(model, model.tokenizer,generated=embed,entry_length=dataset.max_seqs_len[1], temperature=1)[0]

            if out_text.lower()==dataset.answers_raw[item].lower(): 
              acc+=1
            if dataset.answers_raw[item].lower()=='yes' or dataset.answers_raw[item].lower()=='no':
              if out_text.lower()==dataset.answers_raw[item].lower():
                acc_yn+=1
              c_yn+=1
            else:
              if out_text.lower()==dataset.answers_raw[item].lower():
                acc_oe+=1
              c_oe+=1
                
            reference = [str(dataset.answers_raw[item])]
            candidate = [out_text]

            bleu_1 = sentence_bleu(reference[0], candidate[0], weights=(1, 0, 0, 0))

            a = bert_score.compute(references = reference,predictions = candidate,model_type = 'bert-base-uncased')
            bert_avg1+= a['precision'][0]
            bert_avg2+= a['recall'][0]
            bert_avg3+= a['f1'][0]

            
            f1_avg += compute_f1(tokenizer.encode(reference[0]),tokenizer.encode(candidate[0]))
            bleu_avg1+=bleu_1

    
    print('------------')
    print("BLEU {}".format(round(bleu_avg1/len(dataset),3)))
    print("BERTScore {}".format(round(bert_avg3/len(dataset),3)))
    print("F1 {}".format(round(f1_avg/len(dataset),3)))
    print("Accuracy {}".format(round(acc/len(dataset),3)))
    print("Accuracy YN{}".format(round(acc_yn/c_yn,3)))
    print("Accuracy OE{}".format(round(acc_oe/c_oe,3)))

def print_nearest_text_token(vis_token, model):
    """print the nearest token in the vocabulary to the given token through model.gpt.embeddings.weight"""
    embeddings = model.gpt.transformer.wte.weight
    distances = torch.norm(embeddings - vis_token, dim=1)
    nearest_token_idx = torch.argmin(distances)
    print(model.tokenizer.decode([nearest_token_idx.item()]))    
      
def compute_f1(gold_toks, pred_toks):
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1
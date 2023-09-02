import torch
import skimage.io as io
import skimage.transform as transform
import torchvision
import clip
import pandas as pd
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import string
import random
import numpy as np
from transformers import set_seed, GPT2Config, GPT2Tokenizer

def isEglish(s):
    return s.isascii()

def punc(s):
    for c in string.punctuation:
        s=s.replace(c,"")
    return s.lower() 

def update_classes(pkl_train, pkl_val, pkl_test):
    # standardize answer ids across datasets and compute the maximum number of generated output tokens based on the train set
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    with open(pkl_train, 'rb') as f:
            data_train = pickle.load(f)
    with open(pkl_val, 'rb') as f:
            data_val = pickle.load(f)
    with open(pkl_test, 'rb') as f:
            data_test = pickle.load(f)
    
    cur_id = 0
    class_names_list = []
    class_ids_list = [[],[],[]]

    for i, data in enumerate([data_train,data_val,data_test]):
        
        for answer in data['answers']:
            if answer not in class_names_list:
                class_names_list.append(answer)
                class_ids_list[i].append(cur_id)
                cur_id+=1
            else:
                class_ids_list[i].append(class_names_list.index(answer))
    q_lens = []
    a_lens = []
    for question in data_train['questions']:
        q_lens.append(len(tokenizer.encode(question)))
    for answer in data_train['answers']:
        a_lens.append(len(tokenizer.encode(str(answer))))
    
    data_train['class_ids'] = class_ids_list[0]
    data_val['class_ids'] = class_ids_list[1]
    data_test['class_ids'] = class_ids_list[2]
    
    data_train['class_names'] = class_names_list
    data_val['class_names'] = class_names_list
    data_test['class_names'] = class_names_list

    data_train['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    data_val['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    data_test['max_seqs_len']=(int(np.mean(q_lens)+2*np.std(q_lens)),int(np.mean(a_lens)+2*np.std(a_lens)))
    
    with open(pkl_train, 'wb') as f:
        pickle.dump(data_train,f)
    with open(pkl_val, 'wb') as f:
        pickle.dump(data_val,f)
    with open(pkl_test, 'wb') as f:
        pickle.dump(data_test,f)

def preprocess_pathvqa(split, out_path):
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    data =  pd.read_pickle('../vqa_datasets/pathvqa/pathVQAprocessed/split/qas/{}/{}_qa.pkl'.format(split,split))
    print("%0d captions loaded from json " % len(data))
    all_img_prefixes = []
    img_ids = []
    img_paths = []
    all_questions = []
    all_answers = []
    img_dict = {} 
    for i in tqdm(range(len(data))):
        d = data[i]
        if d['answer']!="yes" and d['answer']!="no":
            img_id = d["image"]
            filename = "../vqa_datasets/pathvqa/pathVQAprocessed/split/images/{}/{}.jpg".format(split,img_id)
            with torch.no_grad():
                prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu()                
            if img_id not in img_dict.keys():
                img_dict[img_id] = [[d['question']],[d['answer']],prefix_i,filename]
            else:
                img_dict[img_id][0].append(d['question'])
                img_dict[img_id][1].append(d['answer'])
    # this dictionary is converted into a format that is sutiable for the data loader. Each data point contains a 'img_id', that corresponds is the index of the corresponding
    # CLIP embedding of the image in 'img_prefix'. 
    for img_id, imgs in enumerate(img_dict.keys()):
        all_img_prefixes.append(img_dict[imgs][3])
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q])
            all_answers.append(img_dict[imgs][1][q])
            img_ids.append(img_id)
            img_paths.append(img_dict[imgs][4])

    all_data = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_ids, "questions": all_questions,'answers': all_answers,'img_path': img_paths}

    with open(out_path, 'wb') as f:
        pickle.dump(all_data,f)
    print('Done')
    print("%0d embeddings saved " % len(all_img_prefixes))
def preprocess_ovqa(split, out_path):
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with open('../vqa_datasets/ovqa/{}set.json'.format(split)) as f:
        data =  json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_img_prefixes = []
    img_ids = []
    img_paths = []
    all_questions = []
    all_answers = []
    img_dict = {} 

    # preloading CLIP embeddings for images. Since multiple questions can be associated with one image we construct a dictionary with img ids 
    # as keys for computational efficiency 
    for i in tqdm(range(len(data))):
        d = data[i]
        if isEglish(d['answer']) and isEglish(d['question']):
            img_id = d["image_name"][:-4]
            filename = "../vqa_datasets/ovqa/img/"+d['image_name']
            with torch.no_grad():
                prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu()          
            if img_id not in img_dict.keys():
                img_dict[img_id] = [[punc(d['question'])],[punc(d['answer'])],prefix_i,filename]
            else:
                img_dict[img_id][0].append(punc(d['question']))
                img_dict[img_id][1].append(punc(d['answer']))
    
    # this dictionary is converted into a format that is sutiable for the data loader. Each data point contains a 'img_id', that corresponds is the index of the corresponding
    # CLIP embedding of the image in 'img_prefix'. 
    for img_id, imgs in enumerate(img_dict.keys()):
        all_img_prefixes.append(img_dict[imgs][2])
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q])
            all_answers.append(img_dict[imgs][1][q])
            img_ids.append(img_id)
            img_paths.append(img_dict[imgs][2])

    all_data = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_ids, "questions": all_questions,'answers': all_answers,'img_path': img_paths}

    with open(out_path, 'wb') as f:
        pickle.dump(all_data,f)
    print('Done')
    print("%0d embeddings saved " % len(all_questions))

def preprocess_slake(split, out_path):
    device = torch.device('cuda:0')
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    with open('../vqa_datasets/slake/Slake1.0/{}.json'.format(split)) as f:
        data =  json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_img_prefixes = []
    img_ids = []
    img_paths = []
    all_questions = []
    all_answers = []
    img_dict = {} 
    
    # preloading CLIP embeddings for images. Since multiple questions can be associated with one image we construct a dictionary with img ids 
    # as keys for computational efficiency 
    for i in tqdm(range(len(data))):
        d = data[i]
        if isEglish(d['answer']) and isEglish(d['question']):
            img_id = d["img_id"]
            filename = "../vqa_datasets/slake/Slake1.0/imgs/"+d['img_name']
            with torch.no_grad():
                prefix_i = clip_model.encode_image(preprocess(Image.open(filename)).unsqueeze(0).to(device)).cpu()          
            if img_id not in img_dict.keys():
                img_dict[img_id] = [[d['question']],[d['answer']],prefix_i,filename]
            else:
                img_dict[img_id][0].append(d['question'])
                img_dict[img_id][1].append(d['answer'])
    # this dictionary is converted into a format that is sutiable for the data loader. Each data point contains a 'img_id', that corresponds is the index of the corresponding
    # CLIP embedding of the image in 'img_prefix'. 
    for img_id, imgs in enumerate(img_dict.keys()):
        all_img_prefixes.append(img_dict[imgs][2])
        for q in range(len(img_dict[imgs][0])):
            all_questions.append(img_dict[imgs][0][q])
            all_answers.append(img_dict[imgs][1][q])
            img_ids.append(img_id)
            img_paths.append(img_dict[imgs][2])

    all_data = {"img_prefix": torch.cat(all_img_prefixes, dim=0), "img_ids": img_ids, "questions": all_questions,'answers': all_answers,'img_path': img_paths}

    with open(out_path, 'wb') as f:
        pickle.dump(all_data,f)
    print('Done')
    print("%0d embeddings saved " % len(all_questions))

if __name__=='__main__':
    for split in ['train','test']:
        out_path = "../vqa_datasets/slake/{}.pkl".format(split)
        preprocess_slake(split,out_path)
        update_classes()
    for split in ['train','test','val']:
        out_path = "../vqa_datasets/ovqa/{}.pkl".format(split)
        preprocess_ovqa(split,out_path)
        update_classes()
    for split in ['train','val','test']:
        out_path = "../vqa_datasets/pathvqa/oa_{}.pkl".format(split)
        preprocess_pathvqa(split,out_path)
        update_classes()






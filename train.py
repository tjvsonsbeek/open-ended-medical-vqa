from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from tqdm import tqdm
import copy
import os
import numpy as np
import time
import random
import torch.nn as nn
import torch.nn.functional as nnf
import os
import numpy as np
import random
import pandas as pd
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.nn import functional as nnf
from accelerate import Accelerator
import pdb


def pytorch_model_run(train_loader, valid_loader, model_obj, args):
    ## 💡 using accelerator 
    accelerator = Accelerator()
    device = accelerator.device

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = model_obj.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(train_loader),
    )

    ## 💡 introduce all components to accelerate library
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    valid_loader = accelerator.prepare(valid_loader)

    
    best_valid_loss = float("inf")
    counter = 0
    n_epochs = args.epochs
    accelerator.wait_for_everyone()
    for epoch in range(args.epochs):

        with tqdm(total=args.batch_size * len(train_loader)) as epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")
            start_time = time.time()
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_rocauc = 0.0

            for i, (prefix, labels, tokens, mask, q_len) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    prefix = prefix.type(torch.float32)
                    tokens = tokens.type(torch.long)
                    mask = mask.type(torch.long)
                    q_len = q_len.type(torch.long)
                    outputs = model(prefix, labels, tokens, mask, q_len, batch_size=args.batch_size)
                    logits = outputs.logits
                    loss = 0.

                    shift = 10 if args.setting=="p_tuning" or args.setting=="prompttuning" else 0 

                    for b in range(logits.size(0)):
                        condensed_tokens = tokens[b,q_len[b]+model.prefix_length+1:]
                        condensed_logits = logits[b,shift+q_len[b]+model.prefix_length:-1]

                        loss+= nnf.cross_entropy(condensed_logits.reshape(-1,logits.shape[-1]), condensed_tokens.flatten(), ignore_index=0)
                    loss=loss/logits.size(0)    

                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    avg_loss = total_loss / (i+1)
                    avg_acc = total_acc / (i + 1)
                    avg_roc = total_rocauc / (i + 1)
                    desc = f"Epoch {epoch} - loss {avg_loss:.20f} -accuracy {avg_acc:.4f} -auc {avg_roc:.4f}"
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(prefix.shape[0])
        model.eval()

        total_loss = 0.0
        total_acc = 0.0
        # total_rocauc = 0.0
        with tqdm(total=args.batch_size * len(valid_loader)) as epoch_pbar:
            epoch_pbar.set_description(f"VAL Epoch {epoch}")
            for i, (prefix, labels, tokens, mask,q_len) in enumerate(valid_loader):
                torch.cuda.empty_cache()
                prefix = prefix.type(torch.float32)
                tokens = tokens.type(torch.long)
                mask = mask.type(torch.long)
                q_len = q_len.type(torch.long)

                with torch.no_grad():
                    outputs = model(prefix, labels, tokens, mask, q_len, batch_size=args.batch_size)
                    logits = outputs.logits
                    loss = 0.
                    shift = 10 if args.setting=="p_tuning" or args.setting=="prompttuning" else 0 
                    for b in range(logits.size(0)):
                        condensed_tokens = tokens[b,q_len[b]+model.prefix_length+1:]
                        condensed_logits = logits[b,shift+q_len[b]+model.prefix_length:-1]
                        loss+= nnf.cross_entropy(condensed_logits.reshape(-1,logits.shape[-1]), condensed_tokens.flatten(), ignore_index=0)
                    loss=loss/logits.size(0)    
                    total_loss += loss.item()
                avg_val_loss = total_loss / (i + 1)
                avg_acc = total_acc / (i + 1)
                avg_roc = total_rocauc / (i + 1)
                desc = f"VAL Epoch {epoch} - loss {avg_val_loss:.20f} -acc {avg_acc:.4f} -roc {avg_roc:.4f}"
                epoch_pbar.set_description(desc)
                epoch_pbar.update(prefix.shape[0])

        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss

            torch.save(model.state_dict(), os.path.join(args.out_dir, f"open_ended_latest.pt"))

        scheduler.step()
        elapsed_time = time.time() - start_time
        print(
            "VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s".format(
                epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time
            )
        )
        if avg_val_loss > avg_loss:
            counter += 1
        if counter == 5:
            break
    return model

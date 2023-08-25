import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import argparse

import datetime
from tqdm import tqdm, trange

import os
import sys
sys.path.append("..")
import random

from dataloaders.base_loader import load_data
from algo.DeepFM import DeepFM

from sklearn.metrics import roc_auc_score, log_loss
from utils.config import *

def write_log(w, args):
    os.makedirs(os.path.join('../output', 'logs', args.model, datetime.date.today().strftime('%m%d')), exist_ok=True)
    file_name = os.path.join('../output', 'logs', args.model, datetime.date.today().strftime('%m%d'),
        f"{args.model}_{args.dataset}_LR{args.lr}_WD{args.wd}_BZ{args.batch_size}_DIM{args.dim}_{args.lbd}_{args.num}.log")
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    with open(file_name, 'a') as f:
        f.write(info + '\n')

def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def evaluate_lp(model, data_loader, device, is_test=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for step, (x_user, x_item, x_context, hists, hist_len, label) in enumerate(data_loader):
            x_user, x_item, x_context = x_user.to(device), x_item.to(device), x_context.to(device)
            hists, hist_len = hists.to(device), hist_len.to(device)
            label = label.to(device)

            # model forward
            logits = model(x_user, x_item, x_context, hists, hist_len)
        
            logits = logits.squeeze().detach().cpu().numpy().astype('float64')
            label = label.detach().cpu().numpy()

            predictions.append(logits)
            labels.append(label)

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    auc = roc_auc_score(y_score=predictions, y_true=labels)
    logloss = log_loss(y_true=labels, y_pred=predictions,eps=1e-7, normalize=True)
    return auc, logloss

def main(args):
    #set device and seed
    device = torch.device(f"cuda:{args.gpu}" if (torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    args.device = device
    print(f"Device is {device}.")
    seed_all(args.seed, device)

    #set data and log path
    args.out_path = os.path.join(args.out_path, args.dataset)
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'train_log'), exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'model'), exist_ok=True)
    log_name = args.model+'_'+args.dataset+'_'+args.num+'_seed:'+str(args.seed) + ".txt"
    log_file = open(os.path.join(args.out_path, 'train_log', log_name), "w+")
    log_file.write(str(args))

    #load data
    train_loader, valid_loader, test_loader= load_data(args.dataset, args.batch_size, args.num_workers, args.data_path)

    # model and training components
    args.num_fields = num_fields[args.dataset]
    model = DeepFM(num_feats[args.dataset], num_fields[args.dataset], padding_idxs[args.dataset], args.dim, args.dropout, args).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # start training
    print('Start training.')
    cnt_wait = 0
    best_auc = 0 
    best_t = 0
    
    for epoch in range(args.n_epochs):
        t_loss_list = []
        base_loss_list = []
        infomin_loss_list = []
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f'Epoch: {epoch}/{args.n_epochs}')
            for step, (x_user, x_item, x_context, hists, hist_len, label) in enumerate(train_loader):
                x_user, x_item, x_context = x_user.to(device), x_item.to(device), x_context.to(device)
                hists, hist_len = hists.to(device), hist_len.to(device)
                label = label.to(device)

                # model forward
                logits = model(x_user, x_item, x_context, hists, hist_len)
                feat_dis_loss = model.feature_dis_loss(torch.cat((x_user, x_item, x_context), dim=-1))

                # compute loss
                base_loss = criterion(logits, label.float())
                total_loss = base_loss + args.alpha*feat_dis_loss
                
                base_loss_list.append(base_loss.item())
                infomin_loss_list.append(feat_dis_loss.item())
                t_loss_list.append(total_loss.item())

                # backward
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                t.update()
                t.set_postfix({
                    'train loss': f'{total_loss.item():.4f}',
                    'base loss': f'{base_loss.item():.4f}',
                    'infomin': f'{feat_dis_loss.item():.4f}'
                })
                
            mean_loss = sum(t_loss_list)/len(t_loss_list)
            mean_baseloss = sum(base_loss_list)/len(base_loss_list)
            mean_infomin_loss = sum(infomin_loss_list)/len(infomin_loss_list)

        auc, logloss = evaluate_lp(model, valid_loader, device)
        print(f"Epoch {epoch}: loss: {mean_loss:.6f}, baseloss:{mean_baseloss:.6f}, infomax: {mean_infomax_loss:.6f}, infomin: {mean_infomin_loss:.6f}, club: {mean_club_loss:.6f}\n ")
        print(f"val auc: {auc:.6f}, val logloss: {logloss:6f}\n") 
        write_log(f"Epoch {epoch}: loss: {mean_loss:.6f}, baseloss:{mean_baseloss:.6f}, infomax: {mean_infomax_loss:.6f}, infomin: {mean_infomin_loss:.6f}, club: {mean_club_loss:.6f}\n ", args)
        write_log(f"val auc: {auc:.6f}, val logloss: {logloss:6f}\n", args) 
        log_file.write(f"Epoch {epoch}: loss: {mean_loss:.6f}, baseloss:{mean_baseloss:.6f}, infomax: {mean_infomax_loss:.6f}, infomin: {mean_infomin_loss:.6f}, club: {mean_club_loss:.6f}\n")
        log_file.write(f"val auc: {auc:.6f}, val logloss: {logloss:6f}\n") 
        
        #validate
        if auc > best_auc:
            best_auc = auc
            best_logloss = logloss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.out_path, 'model', f"{args.model}_{args.dataset}_LR{args.lr}_WD{args.wd}_BZ{args.batch_size}_{args.num}"))
            kill_cnt = 0
            write_log("saving model...\n", args)
            log_file.write("saving model...\n")
        else:
            kill_cnt += 1
            if kill_cnt >= args.patience:
                write_log('early stop.\n', args)
                log_file.write('early stop.\n')
                write_log("best epoch: {}".format(best_epoch+1), args)
                log_file.write("best epoch: {}".format(best_epoch+1))
                write_log(f"best AUC: {best_auc:.6f}, best logloss: {best_logloss:.6f}\n", args)
                log_file.write(f"best AUC: {best_auc:.6f}, best logloss: {best_logloss:.6f}\n")
                break
    
    # Test stage.
    model.eval()
    model.load_state_dict(torch.load(os.path.join(args.out_path, 'model', f"{args.model}_{args.dataset}_LR{args.lr}_WD{args.wd}_BZ{args.batch_size}_{args.num}")))
    test_auc, test_logloss = evaluate_lp(model, test_loader, device=device, is_test=True)
    print(f"The test results:\n")
    print(f"test auc: {test_auc}\t test logloss: {test_logloss}\n")
    log_file.write(f"The test results:\n")
    log_file.write(f"test auc: {test_auc}\t test logloss: {test_logloss}\n")
    write_log(f"The test results:\n", args)
    write_log(f"test auc: {test_auc}\t test logloss: {test_logloss}\n", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m', help='which dataset to use')
    parser.add_argument('--model', type=str, default='DeepFM', help='model to use')
    parser.add_argument('--num', type=str, default='0', help='log name')
    parser.add_argument("--data_path", default="../data", help="Path to save the data")
    parser.add_argument("--out_path", default="../output", help="Path to save the output")
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of processes to construct batches")
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--dim', type=int, default=8, help='dimension of entity and relation embeddings')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--ff_dropout', type=float, default=0.0)
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--alpha', type=float, default=0.3, help='weight of infomin.')

    parser.add_argument('--gpu', type=int, default=0, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
        
    parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='patience to stop training')
    parser.add_argument('--hid_units', type=int, default=8, help='neural hidden layer')
    parser.add_argument('--n_neg_max', type=int, default=1, help='the number of negative samples for infomax')
    parser.add_argument('--n_neg_min', type=int, default=1, help='the number of negative samples for informin')
    parser.add_argument('--edge_num', type=int, default=40, help='The number of predicted edges of each data')
    parser.add_argument('--pattern_mode', type=str, choices=['rest', 'att'])
    parser.add_argument('--kg_enc', type=str, choices=['hgcn', 'trans'])


    args = parser.parse_args()

    main(args)
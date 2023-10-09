import torch
import torch.nn as nn


import math
import numpy as np
import argparse
import random
import data_loader as data_loader

from utils import *

import time
from tqdm import tqdm
np.random.seed(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--gpu_num', type=str, default='cpu')
    parser.add_argument('--epochs',type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--L2', default=0.75, type=float)
    parser.add_argument('--savepath',type=str, default='./saved_results/')
    parser.add_argument('--rho', default=0.2, type=float)
    parser.add_argument('--seed', type=int, default = 10)
    parser.add_argument('--lr', type = float, default=0.001)
    parser.add_argument('--datapath', type=str, default='./../SeoulData/urban-core_v2.csv',
                        help='data path')
    parser.add_argument('--split_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--clipgrad',type=int, default = 1000)
    parser.add_argument('--model_name', type=str,default='CNN')
    parser.add_argument('--time_window', type=int,default=6)
    parser.add_argument('--batch_size', type=int,default=256)
    parser.add_argument('--day', type=int, default=0)
    parser.add_argument('--residual', type=int, default = 1)
    parser.add_argument('--cycle', type=int, default = 1)
    parser.add_argument('--cyclepath',type=str, default='./node_idx/core_cycle_1.pkl')

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.model_name == 'CNN_2':
        import CNN_2D as cnn
    else:
        import CNN as cnn

    
    if not args.day:
        dl = data_loader.Data_Loader(args)
        train_set = dl.train_set
        labels = dl.labels
        valid_set = dl.valid_set
    else:
        dl = data_loader.Data_Loader_Day(args)
    
        all_samples = dl.data
        print("Number of sensors",all_samples.shape[0])
        
        dev_sample_index = -1 * int(args.split_percentage * float(all_samples.shape[1]))
        train_set, valid_set = all_samples[:,:,:dev_sample_index], all_samples[:,:,dev_sample_index:]
    # dev_sample_index_valid = int(dev_sample_index*0.75)
    # train_set, valid_set, test_set = all_samples[:,:,:dev_sample_index], all_samples[:,:,dev_sample_index:dev_sample_index_valid], all_samples[:,:,dev_sample_index_valid:]
    if args.residual == 1:
        residual = True
    else:
        residual =False
    model_para = {
        'dilated_channels': 256,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':args.lr,
        'time_window':args.time_window,
        'residual': residual
    }

    if args.gpu_num == 'cpu':
        args.device = 'cpu'
    else:
        args.device = torch.device("cuda:" + str(args.gpu_num) if torch.cuda.is_available() else "cpu")

    model = cnn.Residual_CNN(model_para).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_para['learning_rate'], weight_decay=0)

    criterion = nn.MSELoss()
    len_train = len(train_set)
    count = 0
    best_acc = 0
    INFO_LOG("-------------------------------------------------------train")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        start = time.time()
        if args.day:
            for i in range(train_set.shape[2]-model_para['time_window']):
                inputs = train_set[:,:,:-1][:,:,i:i+model_para['time_window']].to(args.device)
                targets = train_set[:,:,i+model_para['time_window']].to(args.device)
                # inputs, targets = (batch_sam[:, :,:-1]).to(args.device), batch_sam[:,:, 1:].to(
                #     args.device).reshape(-1)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs.squeeze(1).float(), targets.view(-1))
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                if i % max(10, (train_set.shape[2]-model_para['time_window'])//100) == 0:
                # if batch_idx % 1000 == 0:
                    INFO_LOG("epoch: {}\t {}/{}".format(epoch, i, (train_set.shape[2]-model_para['time_window'])))
                    print('Loss: %.3f' % (
                        train_loss / (i + 1)))
        else:
            for i in range(len(train_set)):
                inputs = train_set[i].to(args.device)
                targets = labels[i].to(args.device)
                # inputs, targets = (batch_sam[:, :,:-1]).to(args.device), batch_sam[:,:, 1:].to(
                #     args.device).reshape(-1)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs.squeeze(1).float(), targets.view(-1))
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                if i % max(10, (len(train_set))//100) == 0:
                # if batch_idx % 1000 == 0:
                    INFO_LOG("epoch: {}\t {}/{}".format(epoch, i, (len(train_set))))
                    print('Loss: %.3f' % (
                        train_loss / (i + 1)))
                
    end = time.time()
    model.eval()
    loss_15 = []
    loss_30 = []
    loss_60 = []
    INFO_LOG("-------------------------------------------------------Test")
    with torch.no_grad():
        start = time.time()
        for i in range(12):
            if i ==0:
                if args.day:
                    inputs = train_set[:,:,-model_para['time_window']:].to(args.device)
                    lab = valid_set[:,:,i].view(-1).to(args.device)
                else:
                    inputs = train_set[-1][-1,:,:].unsqueeze(0).to(args.device)
                    lab = valid_set[:,i].to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1).float(),lab).item()
                loss_15.append(loss)
                loss_30.append(loss)
                loss_60.append(loss)
                
                if args.day:
                    inputs = torch.cat((inputs[:,:,1:], outputs.reshape(30,-1,1)),axis=2)
                else:
                    inputs = torch.cat((inputs[:,:,1:], outputs.unsqueeze(0)), axis=2)
            else:
                if args.day:
                    lab = valid_set[:,:,i].view(-1).to(args.device)
                else:
                    lab = valid_set[:,i].to(args.device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1).float(),lab).item()
                if i <3:
                    loss_15.append(loss)
                if i <6:
                    loss_30.append(loss)
                loss_60.append(loss)
                if args.day:
                    inputs = torch.cat((inputs[:,:,1:], outputs.reshape(30,-1,1)),axis=2)
                else:
                    inputs = torch.cat((inputs[:,:,1:], outputs.unsqueeze(0)), axis=2)
    f = open(f'./{args.savepath}_{args.model_name}_day_{args.day}_residual_{args.residual}_cycle_{args.cycle}_cycletype_{args.cyclepath}.txt','a')
    f.write('15 min loss: ' + str(sum(loss_15)/len(loss_15))+'\n')
    f.write('30 min loss: ' + str(sum(loss_30)/len(loss_30))+'\n')
    f.write('60 min loss: ' + str(sum(loss_60)/len(loss_60))+'\n')
    f.close()
    print("15 min loss:",sum(loss_15)/len(loss_15))
    print("30 min loss:",sum(loss_30)/len(loss_30))
    print("60 min loss:",sum(loss_60)/len(loss_60))
    
if __name__ == '__main__':
    main()
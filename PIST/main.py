# import sys
import os
import argparse
import torch
import pandas as pd
import time
from dataloader import *
from utils.utils import *
from model import Mynet
from loss import LabelClusterLoss
from config import *
from utils.metrics import eva

parser = argparse.ArgumentParser()
parser.add_argument('-dataset','--dataset', type=str, default="BeLaE", help='dataset on which the experiment is conducted')
parser.add_argument('-le', '--dim_label_emb', type=int, default=32, help='dimensionality of label embedding')
parser.add_argument('-bs', '--batch_size', type=int, default=512, help='batch size for one iteration during training')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-1, help='learning rate parameter')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='weight decay parameter')
parser.add_argument('-max_epoch', '--max_epoch', type=int, default=500, help='maximal training epochs')
parser.add_argument('-ae', '--att_emb', type=int, default=512, help='dimensionality of label embedding')
parser.add_argument('-wts','--weighting_scheme', type=str, default="average", help='way to summing the marginal distribution')
parser.add_argument('-cuda', '--cuda', action='store_true', help='whether to use gpu')
parser.add_argument('--default_cfg', '-default_cfg', action='store_true', help='whether to run experiment with default hyperparameters')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args,nfold):
    Init_random_seed(seed=0)
    dataset_name = args.dataset
    print(dataset_name)
    dataset = eval(dataset_name)()      #dataset = BeLaE()
    X, Y = dataset.get_data()
    # num_training = Y.size(0)            #number of training examples
    num_dim = Y.size(1)                 #number of dimensions(class variables)
    label_per_dim = {}                  #class labels in each dimension
    num_per_dim = np.zeros((num_dim),dtype = int)  #number of class labels in each dimension
    for dim in range(num_dim):
        labelset = torch.unique(Y[:,dim])
        label_per_dim[dim] = list(labelset)
        num_per_dim[dim] = len(label_per_dim[dim])

    X, Y = X.to(device), Y.to(device)

    configs = generate_default_config()
    configs['dataset'] = dataset
    configs['num_feature'] = X.size(1)
    configs['dim_label_emb'] = args.dim_label_emb         
    configs['att_emb'] = args.att_emb                     
    configs['device'] = device
    configs['weighting_scheme'] = args.weighting_scheme
    configs['weight_decay'] = args.weight_decay
    configs['num_dim'] = num_dim
    configs['label_per_dim'] = label_per_dim
    configs['num_per_dim'] = num_per_dim
    configs['lr'] = args.learning_rate
    # Loading dataset-specific configs
    if args.default_cfg:
        eval('{}_configs'.format(dataset_name))(configs)
    print(configs)

    results = np.zeros((3,10))
    for fold in range(0,nfold):
        best_ham = 0
        log_table = np.zeros(shape=(args.max_epoch, 7))
        results_table = np.zeros(shape=(args.max_epoch, 7))
        train_idx, test_idx = dataset.idx_cv(fold)
        Y_train = Y[train_idx]
        statistics = cooccurrence_statistics(Y_train,configs)
        configs['statistics'] = statistics
        model = Mynet(configs)
        print(model)
        model.reset_parameters()
        
        # set optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=configs['weight_decay'])

        Y_test = Y[test_idx]
        train_iter, test_iter = data_loader(dataset, fold, batch_size=args.batch_size, shuffle=False)
        log_path = "logs/"+args.dataset+"/fold"+str(fold)
        checkpoint_path = "checkpoints/"+args.dataset+"/fold"+str(fold)
        result_path = "results/"+args.dataset+"/fold"+str(fold)
        path_list = [log_path, checkpoint_path, result_path]
        for path in path_list:
            if not os.path.exists(path):
                os.makedirs(path)

        print('Fold'+str(fold)+': start training!')   

        best_epoch = 0
        for epoch in range(args.max_epoch):
            train_loss, emb_loss, cls_loss, pred_Y_train = train(train_iter, model, optimizer, configs)
            train_ham, train_exa, train_sub = eva(Y_train, pred_Y_train)
            if (epoch+1) % 10 == 0:
                print(f"Epoch:{epoch}, train_loss:{train_loss}, emb_loss:{emb_loss}, cls_loss:{cls_loss},\n train_ham:{train_ham}, train_sub:{train_sub}, train_exa:{train_exa}")
            
            log_table[epoch, :] = epoch, train_loss, emb_loss, cls_loss, train_ham.cpu(), train_exa.cpu(), train_sub.cpu()
            np.savetxt(log_path+"/{ds}_le{le}_ae{ae}_lr{lr}_wd{wd}_{wts}_dropout.csv".format(ds=args.dataset, le=configs['dim_label_emb'],ae=configs['att_emb'],lr=configs['lr'], wd=configs['weight_decay'],wts=configs['weighting_scheme']),log_table,delimiter=',', fmt='%1.4f')
            
            test_loss, emb_loss_test, cls_loss_test, pred_Y_test = predict(test_iter, model, configs)
            test_ham, test_exa, test_sub  = eva(Y_test, pred_Y_test)
            test_ham, test_exa, test_sub = test_ham.cpu(), test_exa.cpu(), test_sub.cpu()
            results_table[epoch, :] = epoch, test_loss, emb_loss_test, cls_loss_test, test_ham, test_exa, test_sub 
            np.savetxt(result_path+"/{ds}_le{le}_ae{ae}_lr{lr}_wd{wd}_{wts}_dropout.csv".format(ds=args.dataset, le=configs['dim_label_emb'],ae=configs['att_emb'],lr=configs['lr'], wd=configs['weight_decay'],wts=configs['weighting_scheme']),results_table,delimiter=',', fmt='%1.4f')
            # save model
            if test_ham > best_ham:
                best_ham = test_ham
                best_epoch = epoch
        results[:,fold] = test_ham, test_exa, test_sub
        torch.save({'best_epoch': epoch+1, 'state_dict': model.state_dict()},\
                checkpoint_path+"/{ds}_le{le}_ae{ae}_lr{lr}_wd{wd}_{wts}_dropout.pth".format(ds=args.dataset,le=configs['dim_label_emb'],ae=configs['att_emb'],lr=configs['lr'],wd=configs['weight_decay'],wts=configs['weighting_scheme']))
        print('best epoch:{}'.format(best_epoch))
                
    df = pd.DataFrame(results,index=['hammingscore','exactmatch','subexactmatch'])
    df = df.T
    df.to_csv(result_path[:-6]+"/{ds}_le{le}_ae{ae}_lr{lr}_wd{wd}_{wts}_dropout.csv".format(ds=args.dataset,le=configs['dim_label_emb'],ae=configs['att_emb'],lr=configs['lr'],wd=configs['weight_decay'],wts=configs['weighting_scheme']))


def cooccurrence_statistics(Y,configs):
    '''
    Statistics about co-occurrence probability of two labels w.r.t two dimensions
    Y: tensor
    '''
    num_training = Y.size(0)          #number of training examples
    num_dim = Y.size(1)            #number of dimensions(class variables)
    label_per_dim = configs['label_per_dim']
    num_per_dim = configs['num_per_dim']
    statistics = {}
    for dim0 in range(num_dim):
        statistics[dim0] = {}
        for dim1 in range(dim0,num_dim):
            statistics[dim0][dim1] = torch.zeros(num_per_dim[dim0],num_per_dim[dim1])
            for m in range(num_training):
                for i in range(num_per_dim[dim0]):
                    for j in range(num_per_dim[dim1]):
                        if Y[m][dim0] == label_per_dim[dim0][i] and Y[m][dim1] == label_per_dim[dim1][j]:
                            statistics[dim0][dim1][i][j] += 1
            if dim0 == dim1:
                statistics[dim0][dim1][statistics[dim0][dim1]<1e-6] = 1e-6
            statistics[dim0][dim1] /= num_training

    return statistics

def train(train_iter, model, optimizer, configs):
    model.train()
    train_loss = 0
    emb_loss_item = 0
    cls_loss_item = 0
    pred_Y = []
    criterion = torch.nn.CrossEntropyLoss()
    emb_criterion = LabelClusterLoss()
    for X, Y in train_iter:
        pred_Y_batch = []
        X, Y = X.to(device), Y.to(device)
        num_dim = Y.size(1)
        pairwise_label_emb_list, pred_probs = model(X)

        emb_loss = torch.tensor(0, dtype=torch.float32).to(device)
        for label_emb in pairwise_label_emb_list:
            emb_loss += emb_criterion(label_emb, configs['dim_label_emb'])
        emb_loss /= len(pairwise_label_emb_list)

        cls_loss = torch.tensor(0, dtype=torch.float32).to(device)
        for dim, prob in enumerate(pred_probs):
            cls_loss += criterion(prob,Y[:,dim])
            pred_Y_batch.append(torch.argmax(prob, dim=1, keepdim=True))                  #(n,1)
        cls_loss /= num_dim
        pred_Y_batch = torch.cat(pred_Y_batch,dim=1)                                      #(n,q)
        pred_Y.append(pred_Y_batch)

        loss = cls_loss + emb_loss
        optimizer.zero_grad()    
        loss.backward()          
        optimizer.step()         
        train_loss += loss.item()
        emb_loss_item += emb_loss.item()
        cls_loss_item += cls_loss.item()

    pred_Y = torch.cat(pred_Y, dim=0)

    return train_loss, emb_loss_item, cls_loss_item, pred_Y


def predict(test_iter, model, configs):
    pred_Y = []
    criterion = torch.nn.CrossEntropyLoss()
    emb_criterion = LabelClusterLoss()
    test_loss = 0
    emb_loss_item = 0
    cls_loss_item = 0
    with torch.no_grad():
        model.eval()
        for X, Y in test_iter:
            pred_Y_batch = []
            X, Y = X.to(device), Y.to(device)
            num_dim = Y.size(1)
            pairwise_label_emb_list, pred_probs = model(X)
            emb_loss = torch.tensor(0, dtype=torch.float32).to(device)
            for label_emb in pairwise_label_emb_list:
                emb_loss += emb_criterion(label_emb,configs['dim_label_emb'])
            emb_loss /= len(pairwise_label_emb_list)

            cls_loss = torch.tensor(0, dtype=torch.float32).to(device)
            for dim, prob in enumerate(pred_probs):
                cls_loss += criterion(prob,Y[:,dim])
                pred_Y_batch.append(torch.argmax(prob, dim=1, keepdim=True))                  #(n,1)

            cls_loss /= num_dim
            pred_Y_batch = torch.cat(pred_Y_batch,dim=1)                                      #(n,q)
            pred_Y.append(pred_Y_batch)

            loss = cls_loss + emb_loss
            test_loss += loss.item()
            emb_loss_item += emb_loss.item()
            cls_loss_item += cls_loss.item()            
    
        pred_Y = torch.cat(pred_Y, dim=0)

    return test_loss, emb_loss_item, cls_loss_item, pred_Y


if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()

    main(args,nfold=1)
            
    end_time = time.time()    
    print("during {:.2f}s".format(end_time - start_time))



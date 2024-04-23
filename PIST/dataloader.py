import numpy as np
import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset
import scipy.io as scio
import os

class MDCDataset(Dataset):
    def __init__(self, datadir = "../datasets"):
        datafile = os.path.join(datadir, self.name()+'.mat')
        self.data = scio.loadmat(datafile)
        self.feature = torch.from_numpy(self.data['data'][0][0][1]).type(torch.float)
        self.label = torch.from_numpy(self.data['target']).type(torch.long)-1    
        self.num_dim = self.label.size(1)
        self.num_training = self.label.size(0)
    
    def get_data(self):
        return self.feature, self.label

    def idx_cv(self, fold):
        '''
        fold: 0,1,...,9
        '''
        train_idx = self.data['idx_folds'][fold][0]['train'][0][0].reshape(-1).astype(np.int32)-1
        test_idx = self.data['idx_folds'][fold][0]['test'][0][0].reshape(-1).astype(np.int32)-1
        
        return train_idx, test_idx
    
    def __getitem__(self, index):
        return self.feature[index], self.label[index]
    
    def __len__(self):
        return self.num_training
    
    @classmethod
    def name(cls):
        return cls.__name__

def data_loader(dataset,fold,batch_size,shuffle=True):
    train_idx, test_idx = dataset.idx_cv(fold)
    train_fold = data.dataset.Subset(dataset, train_idx)
    test_fold = data.dataset.Subset(dataset, test_idx)  
    
    train_iter = data.DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=shuffle)
    test_iter = data.DataLoader(dataset=test_fold, batch_size=batch_size, shuffle=shuffle)

    return  train_iter,test_iter

class Adult(MDCDataset):
    pass

class BeLaE(MDCDataset):
    pass

class CoIL2000(MDCDataset):
    pass

class Default(MDCDataset):
    pass

class Flickr(MDCDataset):
    pass

class Scm20d(MDCDataset):
    pass

class TIC2000(MDCDataset):
    pass

class Voice(MDCDataset):
    pass

class WaterQuality(MDCDataset):
    pass

class WQanimals(MDCDataset):
    pass

class WQplants(MDCDataset):
    pass

# if __name__ == '__main__':


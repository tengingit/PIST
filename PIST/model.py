import torch
import torch.nn as nn
from utils.utils import Init_random_seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dropout_layer(X, dropout):
    """
    For pairwise label embedding only.
    """
    assert 0 <= dropout <= 1
    # In this case, all elements are droped.
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are reserved.
    if dropout == 0:
        return X
    mask = (torch.rand(X.size(1)) > dropout).float()
    mask = mask.to(device)
    
    return mask * X / (1.0 - dropout)


class MonoEncoder(nn.Module):
    def __init__(self, configs, dim):
        super(MonoEncoder,self).__init__()
        self.dim_label_emb = configs['dim_label_emb']
        self.dim = dim
        self.num_class = configs['num_per_dim'][dim]
        self.statistics = configs['statistics']
        self.label_emb = nn.Parameter(torch.randn(size=(self.dim_label_emb, self.num_class), requires_grad=True))
        self.device = configs['device']

    def forward(self):
        dim_emb = self.get_pairwise_dimension_embedding()          #(t,1) 

        return self.label_emb, dim_emb        
    
    def get_pairwise_dimension_embedding(self):
        co_occurence_mat = self.statistics[self.dim][self.dim]     #(k,k)
        co_occurence_mat = co_occurence_mat.to(self.device)

        dim_emb = torch.zeros(self.dim_label_emb,1)
        dim_emb = dim_emb.to(self.device)
        emb_mat = self.label_emb                                   #(t,k)
        for i in range(self.num_class):
            if self.training == True:        
                emb_mat = dropout_layer(emb_mat, 0.2)
            dim_emb += torch.sum(co_occurence_mat[i] * emb_mat,dim=1,keepdim=True)

        # pairwise_dim_emb = pairwise_dim_emb.to(self.device)

        return dim_emb


class PairwiseEncoder(nn.Module):
    def __init__(self, configs, dim0, dim1):
        super(PairwiseEncoder,self).__init__()
        self.dim_label_emb = configs['dim_label_emb']
        self.dim0, self.dim1 = dim0, dim1
        self.num_class0 = configs['num_per_dim'][dim0]
        self.num_class1 = configs['num_per_dim'][dim1]
        self.statistics = configs['statistics']
        self.pairwise_label_emb = nn.Parameter(torch.randn(size=(self.dim_label_emb*self.num_class0, self.num_class1), requires_grad=True))
        self.device = configs['device']

    def forward(self):
        pairwise_dim_emb = self.get_pairwise_dimension_embedding()          #(t,1) 

        return self.pairwise_label_emb, pairwise_dim_emb

    def get_pairwise_dimension_embedding(self):
        '''
        For dim0 and dim1, get dimension_embedding with weighted averaging.
        '''
        co_occurence_mat = self.statistics[self.dim0][self.dim1]
        co_occurence_mat = co_occurence_mat.to(self.device)
        num_class0 = co_occurence_mat.size(0)
        # num_class1 = co_occurence_mat.size(1)

        pairwise_dim_emb = torch.zeros(self.dim_label_emb,1)
        pairwise_dim_emb = pairwise_dim_emb.to(self.device)

        for i in range(num_class0):
            emb_mat = self.pairwise_label_emb[i*self.dim_label_emb:(i+1)*self.dim_label_emb,:]
            if self.training == True:
                emb_mat = dropout_layer(emb_mat, 0.2)
            # print(emb_mat.size())
            # print(co_occurence_mat[i].size())
            pairwise_dim_emb += torch.sum(co_occurence_mat[i] * emb_mat,dim=1,keepdim=True)

        # pairwise_dim_emb = pairwise_dim_emb.to(self.device)

        return pairwise_dim_emb

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_list=[], batchNorm=False, dropout=False,
                 nonlinearity='leaky_relu', negative_slope=0.1,
                 with_output_nonlineartity=True):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        self.fcs = nn.ModuleList()
        if hidden_list:
            in_dims = [input_size] + hidden_list
            out_dims = hidden_list + [output_size]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_list):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))  # Controls the angle of the negative slope (which is used for negative input values). Default: 1e-2
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                if dropout:
                    # if i == 0:
                    #     self.fcs.append(nn.Dropout(p=0.2))
                    # elif i < len(hidden_list):
                    if i < len(hidden_list):
                        self.fcs.append(nn.Dropout(p=0.5))
        else:
            self.fcs.append(nn.Linear(input_size, output_size))
            if with_output_nonlineartity:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(output_size, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
                if dropout:
                    self.fcs.append(nn.Dropout())

        self.reset_parameters()
        
    def reset_parameters(self):
        for l in self.fcs:
            # if l.__class__.__name__ == 'Linear':
            #     nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
            #                              nonlinearity=self.nonlinearity)
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)
                # if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                #     nn.init.uniform_(l.bias, 0, 0.1)
                # else:
                #     nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()
    
    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input
    

class DimensionSpecificModel(nn.Module):
    def __init__(self, in_features_x, in_features_y, num_hiddens, output_size,
                 in_layers1=1, out_layers=1, batchNorm=False, dropout=False,
                 nonlinearity='leaky_relu', negative_slope=0.1):
        super(DimensionSpecificModel, self).__init__()
        
        hidden_list = [num_hiddens] * (in_layers1-1)
        self.feature_net = MLP(in_features_x, num_hiddens, hidden_list, batchNorm, dropout, nonlinearity, negative_slope)
        self.attention_net = nn.Linear(in_features_y, num_hiddens)
        
        hidden_list = [num_hiddens] * (out_layers-1)
        self.dim_specific_net = MLP(num_hiddens, output_size, hidden_list, batchNorm, dropout, nonlinearity, negative_slope)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.feature_net.reset_parameters()
        self.dim_specific_net.reset_parameters()
    
    def forward(self, x, y):
        fea = self.feature_net(x) # b1 x h
        atten = self.attention_net(y).sigmoid_() # b2 x h
        specific_fea = fea.unsqueeze(1) * atten.unsqueeze(0) # b1 x b2 x h
        specific_fea = self.dim_specific_net(specific_fea)
        return atten, specific_fea
    
    
class Mynet(nn.Module):
    def __init__(self, configs):
        super(Mynet, self).__init__()
        self.rand_seed = configs['rand_seed']
        self.weighting_scheme = configs['weighting_scheme']
        self.num_dim = configs['num_dim']
        # self.num_pairwise_dim = int(self.num_dim*(self.num_dim-1)/2)
        self.num_pairwise_dim = int(self.num_dim*(self.num_dim+1)/2)
        self.num_per_dim = configs['num_per_dim']
        self.device = configs['device']
        encode_dict = []
        mlp_dict = []

        mapping, _ = self.get_mapping()
        for i in range(self.num_pairwise_dim):
            dim0, dim1 = mapping[i]
            if dim0 == dim1:
                encode_dict.append(MonoEncoder(configs, dim0))
                mlp_dict.append(MLP(configs['att_emb'],self.num_per_dim[dim0],[],batchNorm=False,dropout=True,nonlinearity='relu',with_output_nonlineartity=False))
            else:
                encode_dict.append(PairwiseEncoder(configs, dim0, dim1))
                mlp_dict.append(MLP(configs['att_emb'],self.num_per_dim[dim0]*self.num_per_dim[dim1],[],batchNorm=False,dropout=True,nonlinearity='relu',with_output_nonlineartity=False))

        self.encoders = nn.ModuleList(encode_dict)
        self.mlps = nn.ModuleList(mlp_dict)
        self.DimensionSpecificModel = DimensionSpecificModel(configs['num_feature'],configs['dim_label_emb'],configs['att_emb'],configs['att_emb'],in_layers1=1,
                                                             dropout=True,nonlinearity='relu')

        # self.att_pooling = nn.Linear(configs['att_emb'],1)
      
        # Moving model to the right device for consistent initialization
        self.to(configs['device'])
        
        self.reset_parameters()
        
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        for mlp in self.mlps:
            mlp.reset_parameters()
                
        self.DimensionSpecificModel.reset_parameters()

    def forward(self, X):
        pairwise_dim_emb_list = []
        pairwise_label_emb_list = []
        mapping, inv_mapping = self.get_mapping()

        for i in range(self.num_pairwise_dim):
            dim0, dim1 = mapping[i]
            pairwise_label_emb, pairwise_dim_emb = self.encoders[i]()
            if dim0 != dim1:
                pairwise_label_emb_list.append(pairwise_label_emb)
            pairwise_dim_emb_list.append(pairwise_dim_emb)

        pairwise_dim_emb = torch.cat(pairwise_dim_emb_list,dim=1)                     #(t, q chooses 2)
        pairwise_dim_emb = pairwise_dim_emb.T                                         #(q chooses 2, t)
        atten, dim_specific_emb = self.DimensionSpecificModel(X,pairwise_dim_emb)     #(q chooses 2, d')  (n, q chooses 2, d')

        # atten_weight = self.att_pooling(atten)                                 #(q chooses 2, 1)

        joint_probs = []                                                              
        for i in range(self.num_pairwise_dim):
            dim0, dim1 = mapping[i]
            emb = dim_specific_emb[:,i,:].squeeze(1)                                  #(n, d')
            output = self.mlps[i](emb)                                                #(n, k1k2)
            if dim0 != dim1:
                output = nn.Softmax(dim=1)(output).view(-1,self.num_per_dim[dim0],self.num_per_dim[dim1])        #(n,k1,k2)
            elif dim0 == dim1:
                output = nn.Softmax(dim=1)(output).view(-1,self.num_per_dim[dim0])                               #(n,k1)
            joint_probs.append(output)

        pred_probs = []
        # marginal = []
        for dim in range(self.num_dim):
            marginal_probs = []
            weights = []
            for cond_dim in range(self.num_dim):
                if cond_dim == dim:
                    # continue
                    idx = inv_mapping[dim][dim]
                elif cond_dim < dim:
                    idx = inv_mapping[cond_dim][dim]
                else:
                    idx = inv_mapping[dim][cond_dim]

                joint_prob = joint_probs[idx]                                                      #(n,k1,k2)
                marginal_probs.append(self.sum_prob(dim,cond_dim,joint_prob).unsqueeze(2))         #(n,k,1)
                # weights.append(atten_weight[idx])                                                      
            
            marginal_probs = torch.cat(marginal_probs,dim=2)                                       #(n,k,q)
            # marginal.append(marginal_probs)
            if self.weighting_scheme == 'average':
                pred_probs.append(torch.sum(marginal_probs,dim=2))                    #(n,k)
            # elif self.weighting_scheme == 'attention':
            #     weights = torch.stack(weights,dim=1)                                  #(n,q)
            #     weights = nn.Softmax(dim=1)(weights)                                  #(n,q)
            #     weights = nn.Softmax(dim=1)(weights).unsqueeze(1)                     #(n,1,q)
            #     pred_probs.append(torch.sum(marginal_probs*weights,dim=2))            #(n,k)
            # elif self.weighting_scheme == 'ensemble':
            #     pred_probs.append(self.vote(marginal_probs))
            # elif self.weighting_scheme == 'knn':
            #     pred_probs.append(marginal_probs)

        return pairwise_label_emb_list, pred_probs
    
    def vote(self, marginal_probs):
        indicators = (marginal_probs==torch.max(marginal_probs,dim=1,keepdim=True).values)     #(n,k,q)
        ensemble = torch.sum(indicators,dim=2)                                                 #(n,k)
        pre_results = torch.argmax(ensemble,dim=1)                                             #(k)
        mask = indicators[torch.arange(ensemble.size(0)),pre_results,:].unsqueeze(1)            #(n,1,q)
        right_probs = marginal_probs * mask                                                    #(n,k,q)

        return torch.sum(right_probs,dim=2)                                                    #(n,k)

    def sum_prob(self, dim, cond_dim, joint_prob):
        '''
        return maginal distribution w.r.t dim
        '''
        # assert dim != cond_dim
        if dim < cond_dim:
            assert self.num_per_dim[dim] == joint_prob.size(1)
            marginal_prob = torch.sum(joint_prob,dim=2)                                #(n,k1)
        elif dim > cond_dim:
            assert self.num_per_dim[dim] == joint_prob.size(2)                     
            marginal_prob = torch.sum(joint_prob,dim=1)                                #(n,k2)
        elif dim == cond_dim:
            marginal_prob = joint_prob
        
        return marginal_prob

    def get_mapping(self):
        mapping = []
        inv_mapping = {}
        for i in range(self.num_dim):
            inv_mapping[i] = {}
            for j in range(i,self.num_dim):
                inv_mapping[i][j] = len(mapping)
                mapping.append([i,j])
                                
        return mapping, inv_mapping

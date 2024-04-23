import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LabelClusterLoss(nn.Module):
    def __init__(self):
        super(LabelClusterLoss, self).__init__()
        
    def forward(self, pairwise_label_emb, dim_label_emb):
        '''
        Parameters
        ----------
        pairwise_label_emb : Tensor
            A tk1 x k2 tensor, the embedding of the l_{ij} is stored in emb[(i-1)t:it,j].
        dim_label_emb: int                        
            latent dimension of label embedding (t).
        Returns
        -------
        loss : Tensor
            The link prediction loss.
        '''
        num_class0 = pairwise_label_emb.size(0)//dim_label_emb            #k_1
        num_class1 = pairwise_label_emb.size(1)                           #k_2
        intra_var, inter_var = torch.tensor(0, dtype=torch.float32).to(device), torch.tensor(0, dtype=torch.float32).to(device) 
        dual_intra_var, dual_inter_var = torch.tensor(0, dtype=torch.float32).to(device), torch.tensor(0, dtype=torch.float32).to(device)
        emb_mat_list = []
        dual_emb_mat_list = []

        for i in range(num_class0):
            emb_mat = pairwise_label_emb[i*dim_label_emb:(i+1)*dim_label_emb,:]
            emb_mat_list.append(emb_mat)
            intra_var += self.intra_class_var(emb_mat)
        inter_var = self.inter_class_var(emb_mat_list)

        for j in range(num_class1):
            dual_emb_mat = torch.stack(torch.chunk(pairwise_label_emb[:,j],num_class0),dim=1)
            dual_emb_mat_list.append(dual_emb_mat)
            dual_intra_var += self.intra_class_var(dual_emb_mat)
        dual_inter_var = self.inter_class_var(dual_emb_mat_list)

        loss = 0.5* (intra_var/inter_var + dual_intra_var/dual_inter_var)
        
        return loss/(num_class0*num_class1)
            
    def intra_class_var(self, emb_mat):
        '''
        emb_mat: tensor
            a txk label embedding matrix, the embedding of the l_{ij} is stored in emb_mat[:,j]
        '''
        # dim_label_emb = emb_mat.size(0)           #t
        # num_class = emb_mat.size(1)
        # emb_mat -= emb_mat@torch.ones(num_class)@torch.ones(num_class).t()/num_class          #averge all embeddings
        emb = emb_mat - torch.mean(emb_mat,dim=1,keepdim=True)          #averge all embeddings

        return (emb.t()@emb).trace()

    def inter_class_var(self, emb_mat_list):
        '''
        emb_mat_set: list of tensor
            a list of txk label embedding matrix, the embedding of the l_{ij} is stored in emb_mat[:,j]
        '''
        # dim_label_emb = emb_mat.size(0)           #t
        num_class = emb_mat_list[0].size(1)
        mean_list = []
        for i in range(len(emb_mat_list)):
            mean = torch.mean(emb_mat_list[i],dim=1,keepdim=True)                            #mean embedding
            mean_list.append(mean)

        mean_mat = torch.cat(mean_list,dim=1)
        global_mean = torch.mean(mean_mat,dim=1,keepdim=True)
        global_mat = mean_mat-global_mean

        return num_class*(global_mat.t()@global_mat).trace()
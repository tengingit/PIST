import torch 

def eva(Y,Y_result):
    '''
    Evaluations for MDC.
    '''
    num_testing = Y.shape[0]                     #number of training examples
    num_dim = Y.shape[1]                          #number of dimensions(class variables)
    num_correctdim = torch.sum(Y == Y_result,dim=1)  #number of correct dimmensions for each example
        
    #Hamming Score(or Class Accuracy)
    HammingScore = torch.sum(num_correctdim)/(num_dim*num_testing)    
    
    #Exact Match(or Example Accuracy or Subset Accuracy)
    ExactMatch = torch.sum(num_correctdim == num_dim)/num_testing
    
    #Sub-ExactMatch    
    SubExactMatch = torch.sum(num_correctdim >= num_dim-1)/num_testing

    return HammingScore,ExactMatch,SubExactMatch
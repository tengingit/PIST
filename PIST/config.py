import torch

def generate_default_config():
    configs = {}
    
    # Device
    configs['use_gpu'] = torch.cuda.is_available()
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() and configs['use_gpu'] else 'cpu')
    
    # Dataset
    configs['dataset'] = None
    
    # Training parameters
    configs['lr'] = 1e-3
    configs['weight_decay'] = 1e-4
    configs['batch_size'] = 256
    configs['start_epoch'] = 0
    configs['max_epoch'] = 200

    configs['rand_seed'] = 0
    
    return configs

def Adult_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def BeLaE_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def CoIL2000_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def Default_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512
    
def Flickr_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def Scm20d_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def TIC2000_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def Voice_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def WaterQuality_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def WQanimals_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

def WQplants_configs(configs):
    configs['dim_label_emb'] = 32
    configs['att_emb'] = 512

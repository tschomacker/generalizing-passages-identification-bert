from ml.trainer import Trainer
from ml.multi_label_classification_model import MultiLabelClassificationModel
from ml.custom_dataset import CustomDataset
from ml import label_util

import os
import torch
from torch import cuda
from torch.utils.data import DataLoader
import pandas as pd

def spawn_model(model_config, tokenizer, model_dir, verbose, custom_dataset_key = 'contextualized_clause'):
    """ trains or loads a pytorch-model
    
    Parameters
    ----------
    model_config : dict
        models configuration
    tokenizer : transformers.BertTokenizer
    model_dir : str
        directory were the model should be loaded from or saved to
    custom_dataset_key : str
        used as the key in the customdataset (options are: 'clause' or 'contextualized_clause')
    
    Returns
    -------
    ml.multi_label_classification_model.MultiLabelClassificationModel
        Trained model
    """
    model_config['data']['corpus']
    
    
    epochs_str_for_model_name = 'epochs:'+str(model_config['epochs'])
    if model_config['early_stopping'] is not None:
        epochs_str_for_model_name=epochs_str_for_model_name+'(es_p:'+str(model_config['early_stopping']['patience'])+')'
    
    
    pretrained_str_for_model_name = model_config['data']['pretrained']
    pretrained_str_for_model_name = pretrained_str_for_model_name.split('/')[-1]
    
    model_name = "_".join([model_config['data']['task'],pretrained_str_for_model_name,model_config['data']['corpus'],epochs_str_for_model_name, 
                            model_config['optimizer'], str(model_config['lr']), str(model_config['loss_weights']).replace('_','-'),
                          'dh:'+str(model_config['dropout_hidden']),'da:'+str(model_config['dropout_attention'])
                          ])
    model_path =  os.path.join(model_dir, model_name+'.pt')
    
    print('spawing:',model_path,'on',model_config['device'])
    
    if os.path.exists(model_path):
        # load a model
        print('start loading...')
        trained_model = load_model(model_path, model_config['device'], model_config['data']['pretrained'], model_config['no_labels'])
    else:
        
        print('start training...')
        #train model from scratch
        ## train the model
        experimental_model = MultiLabelClassificationModel(pre_trained_model=model_config['data']['pretrained'], 
                                                            number_of_labels=model_config['no_labels'],
                                                            hidden_dropout_prob=model_config['dropout_hidden'], 
                                                            attention_probs_dropout_prob=model_config['dropout_attention'])
        experimental_model.to(model_config['device'])
        trainer = Trainer()
        
        trained_model, experimental_results = trainer.train(experimental_model, 
                                                model_config['device'],
                                                model_config['epochs'], #epochs 
                                                model_config['optimizer'], #optimizer
                                                model_config['lr'], #lr
                                                model_config['threshold'], #threshold
                                                model_config['data']['train'], 
                                                model_config['data']['validate'],
                                                model_config['loss_func'], # loss function
                                                label_util.generate_loss_weights(model_config['device'], 
                                                                                 [model_config['loss_weights']], 
                                                                                 model_config['data']['train'])[0], # loss function weights
                                                model_config['exclude_none'],
                                                model_config['early_stopping'],
                                                verbose)
        #save to model for later usage
        torch.save(trained_model.state_dict(), model_path)
        print('model was saved under:', model_path)
        
        # prepare the results
        experimental_results['batch_size'] = model_config['data']['batch_size']
        experimental_results['task'] = model_config['data']['task']
        experimental_results['corpus'] = model_config['data']['corpus']
        experimental_results['dropout_hidden'] = model_config['dropout_hidden']
        experimental_results['dropout_attention'] = model_config['dropout_attention']
        
        #save the model config and results in a seperate file
        results_df = pd.DataFrame([experimental_results])
        results_df_file_name = model_path.replace('pt', 'csv')
        results_df.to_csv(results_df_file_name, sep='|', index=False)
        
    return trained_model

def load_model(model_path, device, petrained_model_str, no_labels):
    trained_model = MultiLabelClassificationModel(petrained_model_str, no_labels)
    #try:
    #    trained_model.load_state_dict(torch.load(model_path))
    #    trained_model.to(device)
    #except RuntimeError: 
        # 'Attempting to deserialize object on a CUDA '
    #    print('loading a cuda model on the CPU')
    trained_model.load_state_dict(torch.load(model_path, map_location=torch.device(device)), strict=False)
    trained_model.to(device)
    return trained_model

def create_data_dict(pretrained_model, path, task, tokenizer, max_len, corpus):
    """
    Creates a data dict which will be later used in the experiment parameters
    """
    if 'binary' == task:
        keyword = 'generalization'
    elif 'multi' == task:
        keyword = 'gi'
    else:
        keyword = task
    
    if 'large' in pretrained_model:
        # large models take too much space on the gpu; change depending on hardware setup
        batch_size = 8
    elif 'base' in pretrained_model:
        batch_size = 16
    data_loader_params = {'batch_size': batch_size,
                          'shuffle': False, # NOTE: the dataset is already shuffled during preprocessing
                          'num_workers': 0}
    
    train_set, test_set, validation_set = CustomDataset.from_csv(keyword, tokenizer, max_len,  path, 'contextualized_clause')
    if len(train_set) < 1:
        print(keyword, 'does not seem to be a valid keyword')
    return {'batch_size': data_loader_params['batch_size'], 'pretrained' : pretrained_model,'task' : task, 'corpus' : corpus, 
            'train': DataLoader(train_set, **data_loader_params), 
            'validate': DataLoader(validation_set, **data_loader_params), 'test' : DataLoader(test_set, **data_loader_params)}
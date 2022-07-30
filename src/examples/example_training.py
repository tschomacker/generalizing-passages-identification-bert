import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from transformers import AutoTokenizer
from ml.model_util import create_data_dict, spawn_model
from ml.trainer import Trainer
from torch import cuda

PRETRAINED_MODEL_STR = "deepset/gbert-large"
EVALUATION_TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_STR)
gi_multi_labels = ['none','ALL','BARE','DIV','EXIST','MEIST','NEG']
orignal_gi_multi_labels = ['ALL','BARE','DIV','EXIST','MEIST','NEG']

monaco_path  =  os.path.join('..','..','data','korpus-public.csv' )
example_data_dict = create_data_dict(PRETRAINED_MODEL_STR, monaco_path, 'multi', EVALUATION_TOKENIZER, 206, 'monaco')

example_params = {
    'device' : 'cuda', 
    'epochs': 1, #we used: 20
    'no_labels' : 7,
    'lr' : 1e-04,
    'optimizer' : 'lamb',
    'threshold' : 0.5,
    'data' : example_data_dict,
    'device': 'cuda' if cuda.is_available() else 'cpu', # we strongly discourage trainings on cpu
    'loss_func' :'BCEWithLogitsLoss',
    'dropout_hidden' : 0.3,
    'dropout_attention' : 0.0,
    'exclude_none' : True,
    'loss_weights' : None,
    'early_stopping' : None
}

example_model = spawn_model(example_params, EVALUATION_TOKENIZER, os.path.join('..','..','output','saved_models' ), True)

trainer = Trainer()
example_model_test_results = trainer.test(example_model,
                                         example_params['device'],
                                         example_params['threshold'],
                                         example_params['data']['test'],
                                         example_params['exclude_none'], 
                                         orignal_gi_multi_labels, 
                                         False)
print(example_model_test_results)
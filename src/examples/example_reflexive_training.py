import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from transformers import AutoTokenizer
from ml.model_util import create_data_dict, spawn_model
from ml.trainer import Trainer
from torch import cuda

PRETRAINED_MODEL_STR = "deepset/gbert-large"
EVALUATION_TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_STR)

binary_keyword = 'reflexive_ex_mk_binary'
multi_keyword = 'reflexive_ex_mk_multi'

monaco_path  =  os.path.join('..','..','data','korpus-public.csv' )
example_binary_data_dict = create_data_dict(PRETRAINED_MODEL_STR, monaco_path, binary_keyword, EVALUATION_TOKENIZER, 206, 'monaco')


example_binary_params = {
    'epochs': 20, #we used: 20
    'no_labels' : 1,
    'lr' : 1e-04, 'optimizer' : 'lamb', 'threshold' : 0.5,
    'data' : example_binary_data_dict,
    'device': 'cuda' if cuda.is_available() else 'cpu', # we strongly discourage trainings on cpu
    'loss_func' :'BCEWithLogitsLoss',
    'dropout_hidden' : 0.3, 'dropout_attention' : 0.0,
    'exclude_none' : False,
    'loss_weights' : None, 'early_stopping' : None
}

example_binary_model = spawn_model(example_binary_params, EVALUATION_TOKENIZER, os.path.join('..','..','output','saved_models' ), True)

trainer = Trainer()
example_binary_model_test_results = trainer.test(example_binary_model,
                                         example_binary_params['device'],
                                         example_binary_params['threshold'],
                                         example_binary_params['data']['test'],
                                         example_binary_params['exclude_none'], 
                                         binary_keyword, 
                                         False)
print(example_binary_model_test_results) 

###################################################################################################
# Multi-Label Task

example_multi_data_dict = create_data_dict(PRETRAINED_MODEL_STR, monaco_path, multi_keyword, EVALUATION_TOKENIZER, 206, 'monaco')

example_multi_params = example_binary_params
example_multi_params['data'] = example_multi_data_dict
example_multi_params['no_labels'] = 3

example_multi_model = spawn_model(example_multi_params, EVALUATION_TOKENIZER, os.path.join('..','..','output','saved_models' ), True)

trainer = Trainer()
example_multi_model_test_results = trainer.test(example_multi_model,
                                         example_multi_params['device'],
                                         example_multi_params['threshold'],
                                         example_multi_params['data']['test'],
                                         example_multi_params['exclude_none'], 
                                         multi_keyword, 
                                         False)
print(example_multi_model_test_results) #{'F1-reflexive_ex_mk': 0.8167938931297711, 'F1-macro': 0.7814128195807586, 'F1-micro': 0.7871396895787139}
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

from xai.prediction_pipeline import PredictionPipeline
from xai.lime_clause import LimeClause
from xai.xai_graphic_generator import generate_xai_graphics
from ml.model_util import spawn_model, load_model, create_data_dict
from lime.lime_text import LimeTextExplainer

from transformers import AutoTokenizer
import pandas
from tqdm.auto import tqdm
from torch import cuda


PRETRAINED_MODEL_STR = "deepset/gbert-large"
EVALUATION_TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_STR)
gi_multi_labels = ['none','ALL','BARE','DIV','EXIST','MEIST','NEG']

device = 'cuda' if cuda.is_available() else 'cpu'
print('Using',device,'as device')

# Load the model
cage_small_model = load_model(model_path = os.path.join('..','..','output','saved_models', 
                                                        'multi_gbert-large_cage_small_epochs:20_lamb_0.0001_None_dh:0.3_da:0.0.pt'), 
                              device = device, 
                              petrained_model_str = PRETRAINED_MODEL_STR, 
                              no_labels=7)
# setup XAI
monaco_path  =  os.path.join('..','..','data','korpus-public.csv' )

prediction_pipeline_t_l = PredictionPipeline(cage_small_model, EVALUATION_TOKENIZER, 206, device, gi_multi_labels)
monaco_df = pandas.read_csv(monaco_path , dtype=str, delimiter='|')
lime_clause_list = [LimeClause(monaco_df,'Fontane', '76', '1'), #ALL
               LimeClause(monaco_df,'Goethe', '135', '1'),  #MEIST
               LimeClause(monaco_df,'Goethe', '416', '2'),  #DIV
               LimeClause(monaco_df,'Goethe', '497', '1')]  #NEG
gi_explainer = LimeTextExplainer(class_names=gi_multi_labels, feature_selection='none')

# ATTENTION: Keep in mind, that the number_of_samples highly effects the time it takes to generate a graphic
# So please be patient, when choosing values beyond 5000
# We strongly discourage you from using CPU. 
# It really slows down the whole process. One Graphic can up to 30 minutes to generate
number_of_samples = 5 # default: 5000
coverage_samples = 10 # default: 10000
for lime_clause in tqdm(lime_clause_list, desc='Generate XAI-Grpahic: clause(s)'): 
    generate_xai_graphics(lime_clause, 
                          [prediction_pipeline_t_l],
                          [gi_multi_labels],
                          [gi_explainer],
                          number_of_samples, 
                          coverage_samples,
                          # output directory, where the graphics will be saved
                          os.path.join('..','..','output','test'), 
                          True)
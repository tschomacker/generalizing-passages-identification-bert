# Automatic Identification of Generalizing Passages in German Fictional Texts using BERT with Monolingual and Multilingual Training Data
by [@tschomacker](https://github.com/tschomacker), [@tidoe](https://github.com/tidoe) and [@marina-frick](https://github.com/marina-frick) 
## Introduction
This work is concerned with the automatic identification of generalizing passages like *all ducks lay eggs or tigers are usually striped* (cf. Leslie and Lerner, 2016). In fictional texts, these passages often express some sort of (self-)reflection of a character or narrator or a universal truth which holds across the context of the fictional world (cf. Lahn and Meister, 2016, p. 184), and therefore they are of particular interest for narrative understanding in the computational literary studies.

In the following, we first establish a new state of the art for detecting generalizing passages in German fictional literature using a BERT model (Devlin et al., 2019). In a second step, we test whether the performance can be further improved by adding samples from a non-German corpus to the training data.

We presented our results at the [KONVENS 2022](https://konvens2022.uni-potsdam.de/) Student Poster Session. If you are interested consider:
- Reading our [extended abstract](https://zenodo.org/record/6979859)
- Exploring our [conference poster](https://github.com/tschomacker/???)
- Opening an [Issue](https://github.com/tschomacker/generalizing-passages-identification-bert/issues/new)
- Reaching out to us directly

## :building_construction: Preprocessing
We are using external data and preprocess them to make us of it:
1. Download the [MONACO](https://gitlab.gwdg.de/mona/korpus-public)-data and save it in `data/korpus-public`
1. Download the [SITENT](https://github.com/annefried/sitent/tree/master/annotated_corpus)-data and save it in `data/sitent`
1. switch to src `cd src`
1. Install the requirements via `pip install -r -q requirements.txt`
1. `cd preprocessing`
1. run `python relabel_sitent.py` to create `sitent_gi_labels.json`
1. run `python monaco_preprocessing.py` to create `corpus-public.csv`
1. run `python cage_preprocessing.py` to create `cage.csv` and `cage_small.csv`

Please see Section [Sample format](#sample-format) for more information about the generated data. 

## :test_tube: Experiments
The data created in the previous step needs to be loaded into a DataLoader via a CustomDataset. Our experiments are based on a config-dictionary which includes all parameters. This dict is passed on to the Experiment-Class which executes experimental runs based on the config. Which can be plotted directly. Here is a minimal working example: Experiment configs look like this:

see [examples/example_training.py](https://github.com/tschomacker/generalizing-passages-identification-bert/blob/main/src/examples/example_training.py)
```python
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
```

## Download our models
You can download our [models](https://drive.google.com/drive/folders/119ViOQiT3mYdjBH-QLQ6HR_DCOlOa8nm?usp=sharing) (e.g., in `outpout\saved_models`) and load them.

## XAI
We encourage you to generate your own XAI Graphics with your own clauses or some samples from our dataset. Please download one of our models before.
see: [examples/example_xai.py](https://github.com/tschomacker/generalizing-passages-identification-bert/blob/main/src/examples/example_xai.py)

```python
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
```

## Sample format
The Preprocessing creates the three data set csv-files: MONACO: `corpus-public.csv`, CAGE `cage.csv` and CAGE-small `cage_small.csv`. Each one of them has the following columns:

| **Column**                 | **Description**                                                                                                                | **Example** |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|-------------|
| **clause**                 | original text of the clause                                                                                                    | `ibid.`	            |
| **contextualized clause**  | Clause that is enclosed in `<b> </b>` tags and embedding in its sentence, one previous and one succeeding sentence             | `Boulder: Westview Press. <b> ibid. </b> 2002.` |
| **gi**                     | multi-hot vector with the GI-subtags                                                                                           | 1000000            |
| **generalization**         | 1 or 0, a flag that indicates whether the clause contains generalization                                                       | 1            |
| **document id**            | Text that identifies the clause's document. There are unique and taken from the folder names in MONACO and filenames in SITENT | essays_anth_essay_4.txt             |
| **sent id**                | Unique sentence identifier; continuous integer number                                                                          | 167          |
| **clause id**              | Unique sentence identifier; continuous integer number                                                                          |  1           |
| **instance id**            | Original SITENT-identifier, to map between the corpora                                                                         |  essays_anth_essay_4.txt_310	           |
| **category**               | SITENT-category, MONACO samples do not have a category                                                                         | essays             |
| **dataset**                | train, validate, test                                                                                                          | train            |
| **root_corpus**            | SITENT, MONACO                                                                                                                 | SITENT            |


## Cite Us
```

@conference{schomackerAutomaticIdentificationGeneralizing2022,
	title = {Automatic {Identification} of {Generalizing} {Passages} in {German} {Fictional} {Texts} using {BERT} with {Monolingual} and {Multilingual} {Training} {Data}},
	author = {Schomacker, Thorben and Dönicke, Tillmann and Tropmann-Frick, Marina},
	url = {https://zenodo.org/record/6979859},
	doi = {10.5281/zenodo.6979859},
	abstract = {Extended abstract submitted and accepted for the KONVENS 2022 Student Poster Session. The poster and code are available at https://github.com/tschomacker/generalizing-passages-identification-bert.},
	language = {eng},
	month = sep,
	year = {2022},
	copyright = {Creative Commons Attribution 4.0 International},
	address = {Potsdam, Germany},
}

```
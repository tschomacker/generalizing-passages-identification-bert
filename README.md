# <p align=center>Automatic Identification of Generalizing Passages in German Fictional Texts using BERT with Monolingual and Multilingual Training Data</p>
by [@tschomacker](https://github.com/tschomacker), [@tidoe](https://github.com/tidoe) and [@marina-frick](https://github.com/marina-frick) 
## Introduction
This work is concerned with the automatic identification of generalizing passages like *all ducks lay eggs or tigers are usually striped* (cf. Leslie and Lerner, 2016). In fictional texts, these passages often express some sort of (self-)reflection of a character or narrator or a universal truth which holds across the context of the fictional world (cf. Lahn and Meister, 2016, p. 184), and therefore they are of particular interest for narrative understanding in the computational literary studies.

In the following, we first establish a new state of the art for detecting generalizing passages in German fictional literature using a BERT model (Devlin et al., 2019). In a second step, we test whether the performance can be further improved by adding samples from a non-German corpus to the training data.

We presented our results at the [KONVENS 2022](https://konvens2022.uni-potsdam.de/?page_id=65#studi-poster-session) Student Poster Session. If you are interested consider:
- Reading our [extended abstract](https://zenodo.org/record/6979859)
- Opening an [Issue](https://github.com/tschomacker/generalizing-passages-identification-bert/issues/new)
- Reaching out to us directly

## :building_construction: 1. Preprocessing
We are using external data and preprocess them to make us of it:
1. Download the [MONACO](https://gitlab.gwdg.de/mona/korpus-public)-data and save it in `data/korpus-public`
1. Download the [SITENT](https://github.com/annefried/sitent/tree/master/annotated_corpus)-data and save it in `data/sitent`
1. switch to src `cd src`
1. Install the requirements via `pip install -r -q requirements.txt`
1. `cd preprocessing`
1. run `python relabel_sitent.py` to create `sitent_gi_labels.json`
1. run `python monaco_preprocessing.py` to create `corpus-public.csv`
    - try `python monaco_preprocessing.py -h` in order to see the possibly arguments 
1. run `python cage_preprocessing.py` to create `cage.csv` and `cage_small.csv`

Please see Section [Sample format](#sample-format) for more information about the generated data. 

## :test_tube: 2. Experiments
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

## :inbox_tray: 3. Download our models
### 3.1. Konvens 2022 models
You can download our [models](https://github.com/tschomacker/generalizing-passages-identification-bert/releases/tag/v0.3.0) (e.g., in `outpout\saved_models`) and load them.

For separating Monaco v3.0
```python
python monaco_preprocessing.py \
    --input ../../data/korpus-public-v30 \
    --output ../../data/monaco-v30.csv
```

### 3.2. Reflexivity Classifiers
To create the dataset without Kleist run:
```python
python monaco_preprocessing.py \
    --output ../../data/monaco-ex-kleist.csv \
    --exclude Kleist
```
We trained two separate classification models: 1) on **reflexive_ex_mk_binary** and 2) **reflexive_ex_mk_multi**. The correspondig script is [src/examples/example_reflexive_training.py](https://github.com/tschomacker/generalizing-passages-identification-bert/blob/main/src/examples/example_reflexive_training.py). To learn more about Reflexivity have a look at: 
[Reflexive Passagen und ihre Attribution](https://zenodo.org/record/6328207).

## :medal_sports: 4. Results and Prediction
### 4.1 Results
Besides the experiments in our paper, we additionally evaluated our approach on reflexity classification. The task column is equivalent to the attribute we used from the Table in Section 5.
Table 1: Test results (Truncated after second place after digit)

| Task | F1-binary | F1-micro | F1-macro | F1-GI | F1-Comment | F1-NFR (excl. mk) | Train | Validate Data | Test Data |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generalization | **0.62** | - | - | 0.62 | - | - | Monaco 3.0 without Test, validate | 'Fontane', 'Mann' (Monaco 3.0) | 'Wieland', 'Seghers' (Monaco **4.1**) | 
| generalization | **0.62** | - | - | 0.62 | - | - | Monaco 3.0 without Test, validate | 'Fontane', 'Mann' (Monaco 3.0) | 'Wieland', 'Seghers' (Monaco **3.0**) | 
| generalization | 0.60 | - | - | 0.60 | - | - | CAGE (incl. Monaco 3.0 without Test, validate) | 'Fontane', 'Mann' (Monaco 3.0) | 'Wieland', 'Seghers' (Monaco **3.0**) | 
| generalization | 0.59 | - | - | 0.59 | - | - | CAGE-small (incl. Monaco 3.0 without Test, validate) | 'Fontane', 'Mann' (Monaco 3.0) | 'Wieland', 'Seghers' (Monaco **3.0**) |
| reflexive_ex_mk_binary | 0.69 | - | - | - | - | - | Monaco 4.1 without Test, validate and Kleist | 'Fontane', 'Mann' (Monaco 4.1) | 'Wieland', 'Seghers' (Monaco 4.1) | 
| reflexive_ex_mk_multi | - | 0.64 | 0.65 | **0.62** | 0.68 | 0.62 | Monaco 4.1 without Test, validate and Kleist |'Fontane','Mann' (Monaco 4.1) | 'Wieland', 'Seghers' (Monaco 4.1) |  

### 4.2 Re-Create the results
To quickly re-create our results, download the model and run: 
```
cd src
```
#### 4.2.1 reflexive_ex_mk_binary
```python
python -m ml.test_util --test ../output/saved_models/reflexive_ex_mk_binary_gbert-large_monaco-ex-kleist_epochs:20_lamb_0.0001_None_dh:0.3_da:0.0.pt --data ../data/monaco-ex-kleist.csv --labels reflexive_ex_mk_binary
```

#### 4.2.2 generalization (binary)
Train: MONACO 3.0; Test: MONACO 3.0
```python
python -m ml.saved_model_tester --test ../output/saved_models/binary_gbert-large_monaco_epochs_20_lamb_0.0001_None_dh_0.3_da_0.0.pt \
--data ../data/monaco-v30.csv --labels generalization
```
Train: MONACO 3.0; Test: MONACO 4.1
```python
python -m ml.saved_model_tester --test ../output/saved_models/binary_gbert-large_monaco_epochs_20_lamb_0.0001_None_dh_0.3_da_0.0.pt \
--data ../data/korpus-public.csv --labels generalization
```
Train: Cage; Test: MONACO 3.0 
```python
python -m ml.saved_model_tester --test ../output/saved_models/binary_gbert-large_cage_epochs:20_lamb_0.0001_None_dh:0.3_da:0.0.pt \
--data ../data/monaco-v30.csv --labels generalization
```
Train: Cage-small; Test: MONACO 3.0 
```python
python -m ml.saved_model_tester --test ../output/saved_models/binary_gbert-large_cage_small_epochs:20_lamb_0.0001_None_dh:0.3_da:0.0.pt \
--data ../data/monaco-v30.csv --labels generalization
```

#### 4.2.2
```python
python -m ml.saved_model_tester \
--test ../output/saved_models/reflexive_ex_mk_multi_gbert-large_monaco-ex-kleist_epochs:20_lamb_0.0001_None_dh:0.3_da:0.0.pt \
--data ../data/monaco-ex-kleist.csv \
--labels gi comment nfr  \
--keyword reflexive_ex_mk_multi
```

#### 4.2.2 GI (multi)
```python
python -m ml.saved_model_tester \
--test ../output/saved_models/multi_gbert-large_monaco_epochs:20_lamb_0.0001_None_dh:0.3_da:0.0.pt \
--data ../data/korpus-public.csv \
--labels none ALL BARE DIV EXIST MEIST NEG \
--keyword gi_none
```


```python
python -m ml.saved_model_tester \
--test ../output/saved_models/multi_gbert-large_monaco_epochs:20_lamb_0.0001_None_dh:0.3_da:0.0.pt \
--data ../data/monaco-v30.csv \
--labels none ALL BARE DIV EXIST MEIST NEG \
--keyword gi_none
```




### 4.3 How to create a prediction
To make a predictiction, you can use the following script. The important attributes, that you could change to fit your needs, are: `labels` (Important that the length of this List matches the number of labels/output neurons of the model), `saved_model_path` (download a model as decribed earlier), and `clause` (basically every string is possible, longer strings will be concatenated and using `<b></b>`-tags is recommended)
```python
import os
import sys
#sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from xai.prediction_pipeline import PredictionPipeline
from ml.model_util import spawn_model, load_model, create_data_dict
from transformers import AutoTokenizer
from torch import cuda


PRETRAINED_MODEL_STR = "deepset/gbert-large"
EVALUATION_TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_STR)
labels = ['reflexive']
saved_model_path = os.path.join('..','output','saved_models', 
                                'reflexive_ex_mk_binary_gbert-large_monaco-ex-kleist_epochs_20_lamb_0.0001_None_dh_0.3_da_0.0.pt')

device = 'cuda' if cuda.is_available() else 'cpu'
fine_tuned_model = load_model(model_path = saved_model_path, device = device, 
                              petrained_model_str = PRETRAINED_MODEL_STR, no_labels=len(labels))


prediction_pipeline_t_l = PredictionPipeline(fine_tuned_model, EVALUATION_TOKENIZER, 206, device, labels)
clause = "Aber wenn Leona auch eine vollkommen sachliche Auffassung der sexuellen Frage besa?? , \
        so hatte sie doch auch ihre Romantik . Nur hatte sich bei ihr alles ??berschwengliche , Eitle , \
        Verschwenderische , hatten sich die Gef??hle des Stolzes , des Neides , der Wollust , des Ehrgeizes , \
        der Hingabe, kurz die Triebkr??fte der Pers??nlichkeit und des gesellschaftlichen Aufstiegs durch ein \
        Naturspiel nicht mit dem sogenannten Herzen verbunden,<b> sondern mit dem tractus abdominalis , \
        den E??vorg??ngen,</b> mit denen sie ??brigens in fr??heren Zeiten regelm????ig in Verbindung gestanden \
        sind, was man noch heute an Primitiven oder an breit prassenden Bauern beobachten kann<b> , die \
        Vornehmheit und allerhand anderes,</b> was den Menschen auszeichnet<b> , durch ein Festmahl auszudr??cken \
        verm??gen,</b> bei dem man sich feierlich und mit allen Begleiterscheinungen ??beri??t.An den Tischen ihres Tingeltangels tat Leona ihre Pflicht ;"
prediction = [round(x) for x in prediction_pipeline_t_l.predict(clause)]
print(prediction)
```


## :man_teacher: 5. XAI
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
# It really slows down the whole process. One Graphic can up to 30 minutes to generate on cpu
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

## 5. Sample format
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
| **reflexive_ex_mk_binary** | 1 or 0, a flag that indicates whether the clause contains reflexivity                                                          | 0            |
| **reflexive_ex_mk_multi**  | multi-hot vector for 'gi', 'comment', 'nfr_ex_mk'                                                                              | 010            |

During the pre-processsing one document is split into multiple passage, and each passage into multiple clauses. The attributes of the tables are mostly equivvalent with the attributes defined in the clause-class: [src/preprocessing/models/clause.py](https://github.com/tschomacker/generalizing-passages-identification-bert/blob/main/src/preprocessing/models/clause.py)

## Cite Us
License: [MIT License](LICENSE)
```tex
@conference{schomackerAutomaticIdentificationGeneralizing2022,
	title = {Automatic {Identification} of {Generalizing} {Passages} in {German} {Fictional} {Texts} using {BERT} with {Monolingual} and {Multilingual} {Training} {Data}},
	author = {Schomacker, Thorben and D??nicke, Tillmann and Tropmann-Frick, Marina},
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
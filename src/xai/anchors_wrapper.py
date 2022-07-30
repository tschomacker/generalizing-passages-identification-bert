import spacy
import string

from alibi.explainers import AnchorText
from alibi.utils.download import spacy_model

class AnchorsWrapper:
    """ wrapper that stores all necessary information to generate Anchors
    """
    
    
    def __init__(self, prediction_pipeline_t_l):
        """ 
        Parameters
        ----------
        prediction_pipeline_t_l : 
            Pipeline for predicting the index of the most probable label
        """
    
        
        model = 'de_core_news_sm'
        spacy_model(model=model)
        nlp = spacy.load(model)
        #self.prediction_pipeline_t_l = prediction_pipeline_t_l
        self.explainer = AnchorText(
                predictor=lambda x: prediction_pipeline_t_l.predict_for_anchor(x), 
                sampling_strategy='unknown',  
                nlp=nlp,
                )
    
    
    def generate_anchors(self,sample,
                         coverage_samples = 10000,#default: 10000
                         verbose=False):
        # Anchor Parameters
        # https://docs.seldon.io/projects/alibi/en/v0.5.7/_modules/alibi/explainers/anchor_text.html#AnchorText.explain
        threshold = 0.95 #default: 0.95
        verbose_every = 10 #default: 1
        if verbose:
            print('Anchors for:', sample, '\nwith threshold:',threshold,' coverage_samples:', coverage_samples)
    
        return self.explainer.explain(text=sample, threshold=0.95,coverage_samples=coverage_samples,verbose=verbose)
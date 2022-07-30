from pandas import DataFrame
import numpy as np
from ml.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
from torch import cuda

class PredictionPipeline():
    """
    Wrapper-class to make a model a model callable. __call__ generates 
    predictions for a list of clauses. This behavior is needed for using 
    the model in the LimeTextExplainer.
    """
    def __init__(self, model, tokenizer, max_len, device, class_names):
        self.model= model
        self.tokenizer = tokenizer
        self.max_len=max_len
        self.device=device
        self.class_names = class_names
    
    def __call__(self, clause_list):
        results = []
        for clause in clause_list:
            results.append(self.predict(clause))
        return np.array(results)
    
    def _create_data_loader(self, clause):
        # create a data loader
        ## create a df
        predict_df = DataFrame()
        predict_df['text'] = [clause]
        predict_df['labels'] = [[0]*len(self.class_names)]
        # Create data sets
        predict_set = CustomDataset(predict_df, self.tokenizer, self.max_len)
        # Create data loaders
        predict_params = {'batch_size': 1, 'shuffle': False,'num_workers': 0}
        return DataLoader(predict_set, **predict_params)
        
    def predict(self, clause):
        predict_loader = self._create_data_loader(clause)
        self.model.eval()
        fin_outputs=[]
        with torch.no_grad():
            for data in predict_loader:
                ids = data['ids'].to(self.device, dtype = torch.long)
                mask = data['mask'].to(self.device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                outputs = self.model((ids, mask, token_type_ids))
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        prediction = fin_outputs[0]
        return prediction
    
    def predict_for_anchor(self, list_of_clauses):
        """
        return the index of the label with the highest probability
        """
        prediction_indexes = []
        for clause in list_of_clauses:
            values_per_label = self.predict(clause)
            prediction_indexes.append(values_per_label.index(max(values_per_label)))
            
        return np.asarray(prediction_indexes)
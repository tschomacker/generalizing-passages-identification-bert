import torch
from transformers import BertModel, AutoConfig, AutoModelForMaskedLM, AutoModel

class MultiLabelClassificationModel(torch.nn.Module):
    """
    A Multi-Label Classification based on PyTorch and a Huggingface Transformer Model.
    params:
        pre_trained_model : pre_trained Transformer model. We only tested BertModels
        number_of_labels : is the number of output neurons that are responsible for assigning the labels.
        hidden_dropout_prob: see https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertConfig
        attention_probs_dropout_prob: see https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertConfig
    """
    def __init__(self, pre_trained_model, number_of_labels, hidden_dropout_prob=0.1, attention_probs_dropout_prob= 0.1):
        super(MultiLabelClassificationModel, self).__init__()
        
        # Configure the underlying pretrained model 
        ## Load the corresponding config for the pre-trained model
        pretrained_configuration = AutoConfig.from_pretrained(pre_trained_model)
        ## Change the dropout settings
        pretrained_configuration.hidden_dropout_prob = hidden_dropout_prob
        pretrained_configuration.attention_probs_dropout_prob = attention_probs_dropout_prob
        ## Set it as the first layer in the model
        self.l1 = AutoModel.from_pretrained(pretrained_model_name_or_path=pre_trained_model,
                                                       config=pretrained_configuration)
        # Classification Layer
        ## Input features need to match the pretrained models hidden dimension
        ## large and base models differ in this point
        input_features = pretrained_configuration.hidden_size
        ## Output Features are the labels that should be predicted
        self.output_features = number_of_labels
        self.l2 = torch.nn.Linear(input_features, self.output_features)
        
        # Used to determine the underlying model; for logging results
        self.pre_trained_model = pre_trained_model
    
    def forward(self, x):
        """
        params:
            x
        """
        (ids, attention_mask, token_type_ids) = x
        l1_outputs  = self.l1(ids, attention_mask, token_type_ids)
        output = self.l2(l1_outputs[1])
        return output
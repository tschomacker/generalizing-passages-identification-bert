import torch
import numpy as np
import warnings
from tqdm.auto import tqdm
from ml.lamb import Lamb
from ml.early_stopping import EarlyStopping
from sklearn import metrics


class Trainer():
    """ Utility class that trains and tests a model."""
    
    def prepare_data(device, data):
        """
        prepare data to use it in the model
        """
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        return (ids, mask, token_type_ids)
    
    def train(self, model, device, epochs, optimizer_str, learning_rate, threshold,
              train_loader, test_loader, loss_func, loss_weights, exclude_none, early_stopping_config, verbose):
        """
        Arguments:
            model : MultiLabelClassificationModel
            device : device on which the training will be executed. cpu or cuda
            epochs : int , number of epochs for training and evaluating
            optimizer_str : String that determines which optimizer is used for training.
            learning_rate : float learning_rate during training.
            train_loader : DataLoader for the training data.
            test_loader : DataLoader for the test data.
            criterion (tuple): 
            loss_func: string that indicates which loss function should be used.
            loss_weights: value used as pos_weights or class_weights depend on the selected loss function
                    pos_weights : 
                                a weight of positive examples. Must be a vector with length equal to 
                                the number of classes. If None -> unweighted loss
                                see: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
                    class_weights:
                        see: https://androidkt.com/how-to-use-class-weight-in-crossentropyloss-for-an-imbalanced-dataset/
            exclude_none : see test()
        return : dict with the train and test results
        """
        
        # create the optimizer
        if 'adam' == optimizer_str:
            optimizer = torch.optim.Adam(params =  model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif 'lamb' == optimizer_str:
            optimizer = Lamb(params =  model.parameters(), lr=learning_rate, weight_decay=1e-5)
        else:
            print(optimizer_str,'is not a valid optimizer')
            return None
        
        # intialize the loss function
        if loss_func == 'BCEWithLogitsLoss':
            if loss_weights is not None:
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weights)
            else:
                criterion = torch.nn.BCEWithLogitsLoss()
        elif loss_func == 'CrossEntropyLoss':
            if loss_weights is not None:
                criterion = torch.nn.CrossEntropyLoss(weight=loss_weights,reduction='mean')
            else:
                criterion = torch.nn.CrossEntropyLoss()
        
        mean_train_losses = []
        
        train_test_results = {'pretrained':model.pre_trained_model, 
                                'optimizer': optimizer_str,
                                'lr' : learning_rate,
                                'loss_func':loss_func,
                                'threshold' : threshold}
        
        
        # Start the training and validation
        epoch_count = 0
        
        if early_stopping_config is not None:

            early_stopping = EarlyStopping(patience=early_stopping_config['patience'], 
                                           verbose=early_stopping_config['verbose'], 
                                           delta=early_stopping_config['delta'], 
                                           path= early_stopping_config['path'], 
                                           trace_func=early_stopping_config['trace_func'])
            if verbose:
                print('start using early stopping')
        for epoch in tqdm(range(epochs), desc="Epoch(s)", leave=False):
            epoch_count += 1
            epoch_losses = []
            # train on all batches in the train_loader
            # activate train mode of the model
            model.train()
            for data in tqdm(train_loader, desc="Train Batches", disable= not verbose, leave=False):
                # send to model
                outputs = model(Trainer.prepare_data(device, data))
                # calculate loss
                targets = data['targets'].to(device, dtype = torch.float)
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss))

            mean_train_losses.append(np.mean(epoch_losses))

            # validate the model
            validation_results = self.test(model, device, threshold, test_loader, exclude_none, None , verbose)
            for key in validation_results.keys():
                if key not in train_test_results.keys():
                    train_test_results[key] = []
                train_test_results[key].append(validation_results[key])
            
            if verbose:
                print('epoch:', epoch_count ,'train: loss:', np.mean(epoch_losses), 'validation: ',validation_results)
            
            if early_stopping_config is not None:
                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(validation_results['f1_score_macro'], model)
        
                if early_stopping.early_stop:
                    print("Early stopping at", epoch_count, 'epochs')
                    break
        if loss_weights is None:
            loss_weights_str = '-'
        else:
            loss_weights_str = loss_weights.tolist()
            
        train_test_results['epochs'] = list(range(1,epoch_count+1))
        train_test_results['train_loss'] = mean_train_losses
        train_test_results['loss_weights'] = loss_weights_str
        return model, train_test_results
    
    def test(self, model, device, threshold, testing_loader, exclude_none, labels, verbose):
        """ Used for optimizing/validation and testing of the model.

        Parameters
        ----------
        model
        device
        threshold
        testing_loader
        verbose : 
        exclude_none : bool 
            if the NONE Label should be excluded from the metric calculation

        Returns
        -------
        dict
           Results
        """
        
        activation_func = 'sigmoid' # sigmoid or softmax
        
        if not verbose:
            warnings.filterwarnings("ignore")
                    
        if model.output_features < 2 and exclude_none:
            #raise ValueError('exclude_none can only be True when no_labels > 1')
            pass
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        results =[]
        with torch.no_grad():
            for data in tqdm(testing_loader, desc="Test/Validate Batche(s)", disable= not verbose, leave=False):
                
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(Trainer.prepare_data(device, data))
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                # activation functions
                if 'softmax' == activation_func:
                    fin_outputs.extend(torch.softmax(outputs).cpu().detach().numpy().tolist())
                elif 'sigmoid' == activation_func:
                    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                
            # apply the tst metrics on the model
            outputs = fin_outputs
            
            
            
            targets = fin_targets
            # apply the threshold. Everything below will be 0 and everything above 1
            outputs = np.array(outputs) >= threshold
            outputs = outputs.astype(float)
            outputs = outputs.tolist()

            
            if verbose:
                # print unpredicted to gain insights on the generalization capability of the model
                ## disbalanced predictions
                ## has to be string to be included in a set
                unique_targets_str_set = set([str(label) for label in targets])
                unique_predictions_str_set = set([str(label) for label in outputs])
                print('VALIDATION',
                      '| outputs: sample:',fin_outputs[0][0], 
                      '; max:',max([item for sublist in fin_outputs for item in sublist]), 
                      '; min:',min([item for sublist in fin_outputs for item in sublist]),
                      '| unpredicted labels:',unique_targets_str_set-unique_predictions_str_set
                     )
            
            # exclude the none label
            if exclude_none:
                clean_targets = [target[1:] for target in targets]
                clean_outputs = [output[1:] for output in outputs]
            else:
                clean_targets = targets
                clean_outputs = outputs
            #metrics_dict = ml.multi_label_evaluation.apply_metrics(targets, outputs)
                    
            
            results = {}
            if labels is not None:
                f1_unweighted = metrics.f1_score(y_true=clean_targets, y_pred=clean_outputs, average = None)
                if 1==len(labels):
                    #binary_targets = [num for elem in clean_targets for num in elem]
                    #binary_outputs = [num for elem in clean_outputs for num in elem]
                    
                    results['F1-binary'] = metrics.f1_score(y_true=clean_targets, y_pred=clean_outputs, average = 'binary', pos_label=1.0)
            
                    #print('f1_unweighted',f1_unweighted,
                    #      '\nf1_macro_labels_0', metrics.f1_score(y_true=clean_targets, y_pred=clean_outputs, average = 'macro', labels=[0.0]),
                    #  '\nf1_macro_labels_1', metrics.f1_score(y_true=clean_targets, y_pred=clean_outputs, average = 'macro', labels=[1.0]))
                elif len(labels) != len(f1_unweighted):
                    #warnings.warn('Mismatch: There are '+str(len(labels))+'and'+str(len(f1_unweighted))+'scores')
                    raise ValueError('Mismatch: There are '+str(len(labels))+' labels and '+str(len(f1_unweighted))+' scores')
                else:
                    for label, f1_score_value in zip(labels, f1_unweighted):
                        results['F1-'+label] = f1_score_value
            
            results['F1-unweighted'] = f1_unweighted.tolist()
            results['F1-macro'] = metrics.f1_score(y_true=clean_targets, y_pred=clean_outputs, average = 'macro')
            results['F1-micro'] = metrics.f1_score(y_true=clean_targets, y_pred=clean_outputs, average = 'micro')
            return results
    
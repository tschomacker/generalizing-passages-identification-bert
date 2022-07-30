import torch
from torch import cuda

def generate_loss_weights(device, weights, data_loader=None):
    """
    generates a list of tensors 
        which consists of dimensions times one value of weights
        each tensor is send to device
        the list is later used in the loss function in training
    params:
        device : str that indicates the PyTorch device
        data_loader : DataLoader that contains the data 
        weights : weight mode
            none_scale: 
            neg_scale: 
            None:
            float or int :
    """
    loss_weight_list = []
    none_label_index = 0
    
    if data_loader is not None:
        dimensions = determine_dimensions(data_loader)

    for weight in weights:
        vector = []
        if 'none_scale' == weight:
            weight = none_scale_weights(device, data_loader)
            for i in range(dimensions):
                vector.append(weight)
            vector[none_label_index] = 0
        elif 'neg_scale' == weight:            
            positive_labels, negative_labels = labelwise_pos_neg(data_loader)
            for i in range(dimensions):
                vector.append(float(negative_labels[i]/positive_labels[i]))
        elif isinstance (weight,int) or isinstance (weight,float):
            vector = []
            for i in range(dimensions):
                vector.append(weight)

        if weight is None:
            loss_weight_list.append(None)
        else:
            loss_weight_list.append(torch.FloatTensor(vector).to(device, dtype = torch.float))
    return loss_weight_list

def none_scale_weights(device, data_loader):
    """
    calculates the relation of data points where the first label is 1 to the data points
        where it is 0
    params:
        device : str that indicates the PyTorch device
        data_loader : DataLoader that contains the data 
    return:
        relation : int
    """

    data_points_without_none_label = 0
    data_points_total = 0

    for data in data_loader:
        targets = data['targets'].to(device, dtype = torch.float)
        for label_tensor in targets:
            labels = label_tensor.cpu().detach().numpy().tolist()
            for label in labels:
                if 1 == label:
                    data_points_without_none_label += 1
                data_points_total += 1
    return float(data_points_total/data_points_without_none_label)

def determine_dimensions(data_loader):
    for data in data_loader:
        targets = data['targets'].to('cpu', dtype = torch.float)
        dimensions =  len(targets[0].cpu().detach().numpy().tolist())
        return dimensions
    

def labelwise_pos_neg(data_loader):
    """
    creates two lists:
        positive_labels : list<int>
            Each element in the List represents the number of positive occurences 
            of one label. 
        negative_labels : list<int>
            same as positive_labels for negtive occurences.
        positive_labels[i] + negative_labels[i] = total number of samples
    """
    
    dimensions = determine_dimensions(data_loader)
    positive_labels = [0]*dimensions
    negative_labels = [0]*dimensions
    
    for data in data_loader:
        targets = data['targets'].to('cpu', dtype = torch.float)
        for label_tensor, i in zip(targets, range(dimensions)):
            labels = label_tensor.cpu().detach().numpy().tolist()
            
            for label in labels:
                if 0 == label:
                    negative_labels[i] += 1
                elif 1 == label:
                    positive_labels[i] += 1
                else:
                    print(label,'is an invalid label') 
    return positive_labels, negative_labels

def total_pos_neg_labels(data_loader):
    positive_labels, negative_labels = labelwise_pos_neg(data_loader)
    return sum(positive_labels), sum(negative_labels)
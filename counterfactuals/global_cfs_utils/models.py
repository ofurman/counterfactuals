"""

Contents:

Dictionaries containing dataset names and corresponding
parameters used to train Deep Neural Networks (DNNs):
-normalisers (boolean)
-batch_sizes (int)
-input_output_dims (tuple of ints)
-widths_depths (tuple of ints)
-dropouts (float)
-criteria (torch.nn loss function)
-learning rates (list of floats)
-weight_decays (list of floats)
-num_epochs (list of ints)

Functions:
-performance (assesses model performance on train and test datasets)

Classes:
DNN (Deep Neural Network (DNN) implementation in PyTorch)
DNN_trainer (class for training DNNs in PyTorch)

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as torchdata
import time
from sklearn import metrics

# DNN

# Whether to normalise inputs with mean and standard deviation of training data
dnn_normalisers = {
    'compas': False,
    'german_credit': True,
    'adult_income': True,
    'default_credit': True,
    'heloc': False
}

# Number of inputs back-propogated over per iteration
dnn_batch_sizes = {
    'compas': 128,
    'german_credit': 128,
    'adult_income': 128,
    'default_credit': 128,
    'heloc': 128
}

# Number of inputs and outputs (2 for classification)
dnn_input_output_dims = {
    'compas': (15, 2),         # ~45:55 training label split 0:1
    'german_credit': (71, 2),  # ~30:70 training label split 0:1
    'adult_income': (103, 2),  # ~75:25 training label split 0:1
    'default_credit': (91, 2), # ~22:78 training label split 0:1
    'heloc': (23, 2)           # ~53:47 training label split 0:1
}

# Number of neurons per layer (width) and number of hidden layers (depth)
dnn_widths_depths = {
    'compas': (30, 5),
    'german_credit': (50, 10),
    'adult_income': (50, 5),
    'default_credit': (80, 5),
    'heloc': (50, 5)
}

# Probability of neuron dropout during training
dnn_dropouts = {
    'compas': 0.4,
    'german_credit': 0.6,
    'adult_income': 0.3,
    'default_credit': 0.3,
    'heloc': 0.5
}

# Loss functions
dnn_criteria = {
    'compas': nn.CrossEntropyLoss(),
    'german_credit': nn.CrossEntropyLoss(),
    'adult_income': nn.CrossEntropyLoss(),
    'default_credit': nn.CrossEntropyLoss(weight=torch.tensor([1, 0.5])),
    'heloc': nn.CrossEntropyLoss()
}

# To stagger learning, one can pass multiple values to the following:

# Learning rate of optimizer
dnn_learning_rates = {
    'compas': [1e-3],
    'german_credit': [1e-4, 1e-4],
    'adult_income': [1e-4],
    'default_credit': [1e-4],
    'heloc': [1e-3]
}

# L2 penalty over weights
dnn_weight_decays = {
    'compas': [1e-3],
    'german_credit': [1e-4, 1e-1],
    'adult_income': [1e-4],
    'default_credit': [1e-4],
    'heloc': [1e-4]
}

# Number of training epochs
dnn_num_epochs = {
    'compas': [100],
    'german_credit': [276, 45],
    'adult_income': [50],
    'default_credit': [50],
    'heloc': [109]
}

# XGB

xgb_depths = {
    'compas': 4,
    'german_credit': 6,
    'adult_income': 10,
    'default_credit': 10,
    'heloc': 6
}

xgb_learning_rates = {
    'compas': 0.3,
    'german_credit': 0.01,
    'adult_income': 0.3,
    'default_credit': 0.3,
    'heloc': 0.3
}

xgb_colsamples = {
    'compas': 1,
    'german_credit': 0.3,
    'adult_income': 1,
    'default_credit': 1,
    'heloc': 1
}

xgb_estimators = {
    'compas': 100,
    'german_credit': 500,
    'adult_income': 200,
    'default_credit': 200,
    'heloc': 100
}

xgb_regularizers = {
    'compas': [1, 0, 1],
    'german_credit': [0, 0, 1],
    'adult_income': [2, 4, 1],
    'default_credit': [2, 4, 1],
    'heloc': [4, 4, 1]
}

# LR

lr_normalisers = {
    'compas': False,
    'german_credit': False,
    'adult_income': True,
    'default_credit': True,
    'heloc': False
}

# optional weights for specific datasets/models
lr_class_weights = {
    'compas': None,
    'german_credit': None,
    'adult_income': None,
    'default_credit': {0: 0.65, 1: 0.35},
    'heloc': None
}

lr_max_iters = {
    'compas': 1000,
    'german_credit': 1000,
    'adult_income': 2000,
    'default_credit': 2000,
    'heloc': 3000
}


def performance(model, x_train, x_test, y_train, y_test, normalise=None):
    """
    Function to assess model performance on train and test datasets
    
    Inputs: model (classifier with predict() method)
            x_train, x_test (train and test inputs, respectively)
            y_train, y_test (train and test labels, respectively)
            normalise (if not None, list of train set mean and std devs per feature)
    """
    # Compute predictions on train and test datasets
    if normalise is not None:
        y_pred = model.predict((x_test-normalise[0])/normalise[1])
        y_pred_tr = model.predict((x_train-normalise[0])/normalise[1])
    else:
        y_pred = model.predict(x_test)
        y_pred_tr = model.predict(x_train)

    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    y_pred_tr = y_pred_tr.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    # Compute model accuracy
    print("\033[1mTrain Accuracy:\033[0m {}%"\
          .format(round(metrics.accuracy_score(y_train, y_pred_tr)*100, 2)))
    print("\033[1mTest Accuracy:\033[0m {}%"\
          .format(round(metrics.accuracy_score(y_test, y_pred)*100, 2)))

    # Compute proportion of positive predictions
    print("\033[1mProportion of 1s Predicted (Train):\033[0m {}%"\
          .format(round(sum(y_pred_tr)/len(y_pred_tr)*100, 2)))
    print("\033[1mProportion of 1s Predicted (Test):\033[0m {}%"\
          .format(round(sum(y_pred)/len(y_pred)*100, 2)))


class DNN(nn.Module):
    def __init__(self, input_dim, width, depth, output_dim, dropout=None):
        """
        Class for Deep Neural Network (DNN) implementation in PyTorch
        
        (required arguments)
        input_dim : dimensionality of input data
        width     : width of hidden layers
        depth     : number of layers
        output_dim: number of classes (binary for our recourse application)
        
        (optional argument)
        dropout: probability for neuron dropout
                 (standard approach for regularizing weights)
        """
        # Initialize neurons/layers
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for i in range(depth - 1):
            layers.append(nn.Linear(width, width))
            if dropout:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*layers)
        
        # Softmax used for global translation optimization in self.predict
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Standard forward pass, can be called with DNN(x)
        
        Input: inputs vector
        Output: raw values from output layer
        """
        return self.layers(x)
    
    def predict(self, x, softmax=False):
        """
        Customised prediction function
        
        Input: inputs vector
               softmax (if True, returns softmax predictions)
        Output: hard predictions or softmax values (dependent on softmax argument)
        
        """
        if type(x)!=torch.Tensor:
            x = torch.tensor(x).to(torch.float)
        if softmax:
            ret = self.layers(x)
            return self.sm(ret)
        else:
            ret = self.layers(x)
            _, predicted = torch.max(self.sm(ret), 1)
            return predicted.numpy()
        
class DNN_trainer():
    def __init__(self, model, train_dataset, num_epochs, criterion, optimizer,
                 device, batch_size, val_ratio=None, test_dataset=None):
        """
        Class for Training Deep Neural Networks (PyTorch Implementation)
        
        (required arguments)
        model        : Deep Neural Network model class (models/DNN.py)
        train_dataset: training set (as processed by src/UCI_loader.py)
        num_epochs   : number of training epochs
        criterion    : loss function
        optimizer    : gradient descent/backpropagation optimizer
                       (will contain learning rate)
        device       : 'cpu' or 'cuda' for CPU or GPU respectively
        batch_size   : batch_size
        
        (optional arguments)
        val_ratio    : proportion of dataset used for validation set
                       (if None, will not generate a validation
                        set for early stopping)
        test_loader  : test set (to assist debugging of early stop method)
        """
        # Dataloaders and Validation Set
        self.val_ratio = val_ratio
        if self.val_ratio:
            train_set_size = int(len(train_dataset) * (1-self.val_ratio))
            val_set_size = len(train_dataset) - train_set_size
            train_dataset, val_dataset =\
            torchdata.random_split(train_dataset, [train_set_size, val_set_size])
            self.val_loader = torchdata.DataLoader(dataset=val_dataset, 
                                                   batch_size=batch_size,
                                                   shuffle=False)
        self.train_loader = torchdata.DataLoader(dataset=train_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
        if test_dataset is not None:
            self.test_loader = torchdata.DataLoader(dataset=test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)
        else:
            self.test_loader = None
        
        # Initialises losses and optimization parameters
        self.losses = np.zeros(num_epochs)
        self.losses[:] = np.nan
        self.val_losses = np.zeros(num_epochs)
        self.val_losses[:] = np.nan
        if self.test_loader:
            self.test_losses = np.zeros(num_epochs)
            self.test_losses[:] = np.nan
        self.total_step = len(self.train_loader)
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        
    def train(self, num_epochs, patience, print_outputs=False):
        """
        Main function call for DNN gradient descent training
        
        Inputs: num_epochs (total number of training epochs)
              : patience (standard early stop patience parameter in epochs)
              : print_outputs (if True, print training progression continually)
        Output: trained model
        """
        print('Number of Epochs = {}'.format(num_epochs))
        print('Number of Steps per Epoch = ceil(Length of Trainset / Batch Size) = {}'\
              .format(self.total_step))
        self.start_time = time.time()
        self.num_epochs = num_epochs
        self.print_outputs = print_outputs
        early_stop_count = 0
        last_val_loss = np.inf
        
        for self.epoch in range(self.num_epochs):
            # Training loss
            loss = self.train_epoch()
            self.losses[self.epoch] = loss
            train_acc = self.test(self.train_loader)
            print('Accuracy of the network on the training set: {}%'.format(train_acc))
            test_acc = self.test(self.test_loader)
            print('Accuracy of the network on the test set: {}%'.format(test_acc))
            
            # Validation loss
            if self.val_ratio:
                val_loss = self.validation_epoch(self.val_loader)
                self.val_losses[self.epoch] = val_loss
                if val_loss > last_val_loss:
                    early_stop_count += 1
                    if early_stop_count >= patience:
                        print('Early Stopping at Epoch:', self.epoch)
                        if self.test_loader:
                            self.test_losses[self.epoch] =\
                            self.validation_epoch(self.test_loader)
                            self.plot_progress(self.losses[:self.epoch+1],
                                               self.val_losses[:self.epoch+1],
                                               self.test_losses[:self.epoch+1])
                        else:
                            self.plot_progress(self.losses[:self.epoch+1],
                                               self.val_losses[:self.epoch+1])
                        return self.model
                else:
                    early_stop_count = 0
                last_val_loss = val_loss
            
            # Test loss (debugging early stop)
            if self.test_loader:
                self.test_losses[self.epoch] = self.validation_epoch(self.test_loader)
        if self.val_ratio and self.test_loader:
            self.plot_progress(self.losses, self.val_losses, self.test_losses)
        elif self.val_ratio:
            self.plot_progress(self.losses, self.val_losses)
        elif self.test_loader:
            self.plot_progress(self.losses, test_losses=self.test_losses)
        else:
            self.plot_progress(self.losses)
        return self.model
            
    def train_epoch(self):
        """
        Gradient descent operation computed during training
        
        Output: average training loss value
        """
        self.model.train()
        loss_total = 0
        total = 0
        for i, (inputs, labels) in enumerate(self.train_loader):
            # Move tensors to the configured device
            inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
            labels = torch.stack((1-labels, labels), dim=1).to(self.device)
            #labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss_total += loss.item()
            total += labels.size(0)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print step
            t = time.time() - self.start_time
            if self.print_outputs:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.2f}s'\
                       .format(self.epoch+1, self.num_epochs, i+1,
                               self.total_step, loss_total, t), end='\r')
        if self.print_outputs:
            print()
        return loss_total / total

    def validation_epoch(self, loader):
        """
        Validation loss computation computed per epoch of training
        
        Input: torchvision validation set loader
        Output: average validation loss value
        """
        self.model.eval()
        loss_total = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                # Move tensors to the configured device
                inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
                labels = labels.to(self.device)
                labels = torch.stack((1-labels, labels), dim=1).to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item()
                total += labels.size(0)

            # Print step
            t = time.time() - self.start_time
        if self.print_outputs and self.val_ratio:
            print('Validation Loss: {:.4f}, Time: {:.2f}s'.format(loss.item(), t))
        return loss_total / total
    
    def test(self, test_loader):
        """
        Test set performance computed post-training
        
        Input: torchvision test set loader
        Output: test set accuracy
        """
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            props_1 = 0
            for inputs, labels in test_loader:
                inputs = inputs.reshape(inputs.shape[0], -1).to(self.device)
                labels = (labels>0).to(int).to(self.device)  # binary classification
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                props_1 += predicted.sum().item()
            print('Proportion of 1\'s predicted: {:.3f}%'.format(100 * props_1 / total))
            return 100 * correct / total
    
    def plot_progress(self, losses, val_losses=None, test_losses=None):
        """
        Training, validation and test losses plotted post-training
        
        Inputs: losses (training set)
              : val_losses (validation set)
              : test_losses (test set)
        """
        plt.figure(figsize=(5.5,3), dpi=200)
        plt.plot(losses, label='Training Loss')
        if val_losses is not None:
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
        if test_losses is not None:
            plt.plot(test_losses, label='Test Loss')
            plt.legend()
        plt.title('Model Training: Loss vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

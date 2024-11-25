import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
#import torchvision.transforms.functional as F
import tqdm
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                indices.append(art)
                classes.add(arttype)
        print("...finished")
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = list(classes)
        self.device = device
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        return image, ilabel

class WikiArtModel(nn.Module):
    def __init__(self, hidden_size=300, output_size=3, optimizer='Adam', learning_rate=0.001, loss_function='CEL', l1=0.0, l2=0.0, scheduler=None, param_initialisation=None):
        super(WikiArtModel, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        #batchnorm takes care of the sizes of the images? 
        self.batchnorm1d = nn.BatchNorm1d(105*105)
        self.linear1 = nn.Linear(105*105, hidden_size)
        self.dropout = nn.Dropout(0.01)
        self.relu4 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

        
        #different weight initialisations can be chosen for each of the layers. 
        #Here we are going with the Default initialisation. 
        self._init_weights = 'Default'

        # Initialize optimizer, loss function, and other parameters
        self.optimizer = self.get_optimizer(optimizer, learning_rate, l2)
        self.loss = self.get_loss(loss_function)
        self.learning_rate = learning_rate
        self.l1_rate = l1
        self.l2_rate = l2

        self.scheduler = self.get_scheduler(scheduler, self.optimizer)
        
        if param_initialisation is not None: 
            layer, init_method = param_initialisation 
            self.initialize_weights(layer, init_method)
    
    def forward(self, x):
        # Forward pass through the convolutional layers
        #Allocate the tensors to the specified device
        x = x.to(next(self.parameters()).device)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Flatten the tensor
        x = x.view(-1, 64 * 5 * 5)

        # Forward pass through the fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        #Since CEL expects logits we won't apply the softmax here.
        #x = self.log_softmax(x)

        return x

    #this method takes a single image as input and returns either the predicted label or the probability distribution of  all labels. 
    def predict(self, image, return_probabilities = False):

        self.eval()
        image = image.to(next(self.parameters()).device)

        # Ensure input tensor is in the correct shape (batch_size=1, channels=1, height, width)
        if len(image.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.forward(input_tensor)
        
        if return_probabilities:
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)
            return probabilities
        
        _, predicted_class = torch.max(output, dim=1)

        return predicted_class.item()

    def get_optimizer(self, optimizer, learning_rate, l2):
        Optimizers = {
            'Adam': optim.Adam(self.parameters(), lr=learning_rate, weight_decay=l2), 
            'SGD': optim.SGD(self.parameters(), lr=learning_rate, momentum=0.09, weight_decay=l2)
        }
        return Optimizers[optimizer]

    def l1_regularization(self, loss):
        l1_reg = sum(p.abs().sum() * self.l1_rate for p in self.parameters())
        loss += l1_reg
        return loss

    def get_loss(self, loss_function):
        Loss_Functions = {
            'CEL': nn.CrossEntropyLoss(), 
        }
        return Loss_Functions[loss_function]
    
    def get_scheduler(self, scheduler, optimizer):
        if scheduler is None:
            return None
        schedulers = {
            'OnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.00001)
        }
        return schedulers[scheduler]

    def initialize_weights(self, layer_init=None, initialisation='Normal'):
        init_methods = {
            'Xavier': init.xavier_uniform_, 
            'Uniform': lambda x: init.uniform_(x, a=-0.1, b=0.1), 
            'Normal': lambda x: init.normal_(x, mean=0, std=0.01), 
            'He': lambda x: init.kaiming_normal_(x, mode='fan_in', nonlinearity='relu')
        }

        self._init_weights = init_methods[initialisation]

        if layer_init is None:
            print('no layer specified')
            parameters = self.named_parameters()
            print(f'{initialisation} initialization for all weights')
        else: 
            parameters = layer_init.named_parameters()
            print(f'{initialisation} initialization for {layer_init}')

        for name, param in parameters:
            if 'weight' in name:
                self._init_weights(param)
            elif 'bias' in name:
                # Initialize biases to zeros
                nn.init.constant_(param, 0)





'''
class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv2d = nn.Conv2d(3, 1, (4,4), padding=2)
        self.maxpool2d = nn.MaxPool2d((4,4), padding=2)
        self.flatten = nn.Flatten()
        self.batchnorm1d = nn.BatchNorm1d(105*105)
        self.linear1 = nn.Linear(105*105, 300)
        self.dropout = nn.Dropout(0.01)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(300, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        #print("convout {}".format(output.size()))
        output = self.maxpool2d(output)
        #print("poolout {}".format(output.size()))        
        output = self.flatten(output)
        output = self.batchnorm1d(output)
        #print("poolout {}".format(output.size()))        
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.relu(output)
        output = self.linear2(output)
        return self.softmax(output)

'''

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtModel, CNNWikiArt
import json
import argparse

import Trainer

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file", default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
device = torch.cuda.get_device_name(torch.cuda.current_device())


print("Running...")


traindataset = WikiArtDataset(trainingdir, device)
#testingdataset = WikiArtDataset(testingdir, device)

print(traindataset.imgdir)

the_image, the_label = traindataset[5]
print(the_image, the_image.size())

# the_showable_image = F.to_pil_image(the_image)
# print("Label of img 5 is {}".format(the_label))
# the_showable_image.show()

num_of_labels = len(traindataset.classes)



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(data, labels):
    train_data = Subset(dataset, train_idx)
    test_datas = Subset(dataset, test_idx)






hyperparameter_optimization = False
#Train the model with an option of hyperparameter optimization
if hyperparameter_optimization == True:
    print('Hyperparameter Optimization Loop')
    best_params, best_accuracy = Trainer.hyperparameter_optimization(train_data, val_data,  num_of_labels, device, n_trials = 15)
    cnn_model= CNNWikiArt(best_params['hidden_size'], num_of_labels, optimizer = best_params['optimizer'], learning_rate = best_params['learning_rate']).to(device)
    trainer = Trainer(50, best_params['batch_size'], early_stopping_patience = 7)
    trainer.fit(cnn_model, train_data, val_data)

else: 
    cnn_model = CNNWikiArt(300, output_size=num_of_labels, optimizer = 'Adam', learning_rate = 0.001, l2 = 0.0).to(device)
    #print(next(cnn_model.parameters()).device)
    trainer = Trainer(max_epochs = config["epochs"], batch_size = config["batch_size"])
    trainer.fit(cnn_model,train_data,val_data)

#Plot Progress report
n_epochs = range(trainer.max_epochs)
train_loss = trainer.train_loss_values
nan_values = np.full(trainer.max_epochs - len(train_loss), np.nan)
train_loss = np.concatenate([train_loss,nan_values])

val_loss = trainer.val_loss_values
nan_values = np.full(trainer.max_epochs - len(val_loss), np.nan)
val_loss = np.concatenate([val_loss,nan_values])

plt.figure(figsize=(10,6))
plt.plot(n_epochs, train_loss, color='blue', label='train_loss' , linestyle='-')
plt.plot(n_epochs, val_loss, color='orange', label='val_loss' , linestyle='-')
plt.title("Train Loss")
plt.legend()

#################################################################
'''
#################################################################
#Test the model and save it
#################################################################  

trainer.test(cnn_model, test_data)

torch.save(cnn_model.state_dict(), 'model_parameters.pth')
torch.save(cnn_model, 'model.pth')
'''

'''
def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

    model = WikiArtModel().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss().to(device)
    
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(config["epochs"], config["batch_size"], modelfile=config["modelfile"], device=device)

'''

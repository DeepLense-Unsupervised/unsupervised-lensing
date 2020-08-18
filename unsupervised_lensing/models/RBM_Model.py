import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd
import sys
from .RBM_Nets import *

def train(data_path='./Data/no_sub_train.npy',
          epochs=50,
          learning_rate=2e-3,
          optimizer='Adam',
          checkpoint_path='./Weights',
          pretrain=True,
          pretrain_mode='transfer',
          pretrain_model='A'):
        
        '''
        
        Args:
        ______
        
        data_path: str
           Path to your input NumPy array of shape [number_of_batches, batch_size, number_of_channels, height, width]
                      
        epochs: int
        
        learning_rate: float
        
        optimizer: str
            Choose Optimizer for training the model, available options: ['Adam', 'RMSprop', 'SGD']
            
        checkpoint_path: str
            Path to store model weights
        
        pretrain: bool
            Will continue training from preloaded weights if set to True
            
        pretrain_mode: str
            
            'transfer': Will load the pre-trained model weights from Google Drive
            'continue': Will load the model weights from the 'checkpoint_path' directory
            
        pretrain_model: str ('A','B')
            Select the model for loading the weights when 'pretrain_mode' is set to transfer. Refer [paper link]
        '''
        
        x_train = np.load(data_path)
        x_train = x_train.astype(np.float32)
        print('Data Imported')
        
        if x_train.shape[2] != 1:
            print('RBM model only supports single channel data')
            sys.exit()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RBM().to(device)

        if pretrain == True:
        
            if pretrain_mode == 'transfer':
            
                print('Downloading Pretrained Model Weights')
                if pretrain_model == 'A':
                    gdd.download_file_from_google_drive(file_id='10NnzEh-iW-y540D0Pzkuzz9G_Rw-Aabr', dest_path=checkpoint_path + '/RBM.pth')
                else:
                    gdd.download_file_from_google_drive(file_id='1rMmgk60jT9Zr58S-81CNSiEmWDv0pKiP', dest_path=checkpoint_path + '/RBM.pth')
                    
                if torch.cuda.is_available():
                    model = torch.load(checkpoint_path + '/RBM.pth')
                else:
                    model = torch.load(checkpoint_path + '/RBM.pth', map_location=torch.device('cpu'))
                
            if pretrain_mode == 'continue':
            
                print('Importing Pretrained Model Weights')
                if torch.cuda.is_available():
                    model = torch.load(checkpoint_path + '/RBM.pth')
                else:
                    model = torch.load(checkpoint_path + '/RBM.pth', map_location=torch.device('cpu'))

        criteria = nn.MSELoss()
        
        if optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        n_epochs = epochs
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=n_epochs, steps_per_epoch=x_train.shape[0])

        print('Training the model!')
        loss_array = []
        for epoch in tqdm(range(1, n_epochs+1)):
            train_loss = 0.0

            for i in range(x_train.shape[0]):
                data = torch.from_numpy(x_train[i])
                if torch.cuda.is_available():
                  data = data.cuda()
                optimizer.zero_grad()
                v, v_gibbs, hidden = model(data.view(-1, 22500))
                loss = model.free_energy(v) - model.free_energy(v_gibbs)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()*data.size(0)

            train_loss = train_loss/x_train.shape[0]
            loss_array.append(train_loss)

            torch.save(model, checkpoint_path + '/RBM.pth')

        return loss_array
        
def evaluate(data_path='./Data/no_sub_test.npy',
          checkpoint_path='./Weights',
          out_path='./Results',
          pretrain_mode='transfer',
          pretrain_model='A'):
          
        '''

        Args:
        ______

        data_path: str
            Path to your input NumPy array of shape [number_of_batches, batch_size, number_of_channels, height, width]
                              
        checkpoint_path: str
            Path to store model weights
          
        out_path: str
            Path to store reconstructed lenses
          
        pretrain_mode: str
          
            'transfer': Will load the pre-trained model weights from Google Drive
            'continue': Will load the model weights from the 'checkpoint_path' directory
          
        pretrain_model: str ('A','B')
            Select the model for loading the weights when 'pretrain_mode' is set to transfer. Refer [paper link]
        '''
        
        x_train = np.load(data_path)
        train_data = x_train.astype(np.float32)
        print('Data Imported')
        
        if x_train.shape[2] != 1:
            print('RBM model only supports single channel data')
            sys.exit()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RBM().to(device)
        
        if pretrain_mode == 'transfer':
        
            print('Downloading Pretrained Model Weights')
            if pretrain_model == 'A':
                gdd.download_file_from_google_drive(file_id='10NnzEh-iW-y540D0Pzkuzz9G_Rw-Aabr', dest_path=checkpoint_path + '/RBM.pth')
            else:
                gdd.download_file_from_google_drive(file_id='1rMmgk60jT9Zr58S-81CNSiEmWDv0pKiP', dest_path=checkpoint_path + '/RBM.pth')
                
            if torch.cuda.is_available():
                model = torch.load(checkpoint_path + '/RBM.pth')
            else:
                model = torch.load(checkpoint_path + '/RBM.pth', map_location=torch.device('cpu'))
            
        if pretrain_mode == 'continue':
        
            print('Importing Pretrained Model Weights')
            if torch.cuda.is_available():
                model = torch.load(checkpoint_path + '/RBM.pth')
            else:
                model = torch.load(checkpoint_path + '/RBM.pth', map_location=torch.device('cpu'))
                    
        criteria = nn.MSELoss()
        out = []
        for i in tqdm(range(train_data.shape[0])):
          data = torch.from_numpy(train_data[i])
          if torch.cuda.is_available():
            data = data.cuda()
          v, v_gibbs, hidden = model(data.view(-1, 22500))
          v_gibbs = v_gibbs.view(-1,1,150,150)
          out.append(v_gibbs.cpu().detach().numpy())
        out = np.asarray(out)

        output = []
        for i in range(out.shape[0]):
          for j in range(out[i].shape[0]):
            output.append(out[i][j])

        output = np.asarray(output)
        np.save(out_path + '/Recon_samples.npy',output)
        
        temp1 = []
        for i in range(train_data.shape[0]):
          for j in range(train_data[i].shape[0]):
            temp1.append(train_data[i][j])
        train_data = np.asarray(temp1)
        
        losses = []
        for i in range(train_data.shape[0]):
            losses.append(criteria(torch.from_numpy(train_data[i]), torch.from_numpy(output[i])))
        
        return np.asarray(losses)

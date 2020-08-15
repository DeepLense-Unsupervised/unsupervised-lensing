import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd

class DCA(nn.Module):
    def __init__(self):
        super(DCA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.Flatten(),
            nn.Linear(5184, 1000),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 5184)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 7, stride=3, padding=1, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 6, stride=3, padding=1, output_padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(-1,64,9,9)
        x = self.decoder(x)
        return x

def train(data_path='./Data/no_sub_train.npy',
          epochs=50,
          learning_rate=2e-3,
          optimizer='Adam',
          checkpoint_path='./Weights',
          pretrain=False,
          pretrain_mode='transfer',
          pretrain_model='A'):
          
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DCA().to(device)

        if pretrain == True:
        
            if pretrain_mode == 'transfer':
            
                print('Downloading Pretrained Model Weights')
                if pretrain_model == 'A':
                    gdd.download_file_from_google_drive(file_id='1GzJtcdvMXL7Py9NGCYqkMGQbOrDTtH_U', dest_path=checkpoint_path + '/DCAE.pth')
                else:
                    gdd.download_file_from_google_drive(file_id='1jjyMGjr6KGPXjblwRygBlV6YxtsmRkzl', dest_path=checkpoint_path + '/DCAE.pth')
                    
                if torch.cuda.is_available():
                    model = torch.load(checkpoint_path + '/DCAE.pth')
                else:
                    model = torch.load(checkpoint_path + '/DCAE.pth', map_location=torch.device('cpu'))
                
            if pretrain_mode == 'continue':
            
                print('Importing Pretrained Model Weights')
                if torch.cuda.is_available():
                    model = torch.load(checkpoint_path + '/DCAE.pth')
                else:
                    model = torch.load(checkpoint_path + '/DCAE.pth', map_location=torch.device('cpu'))

        x_train = np.load(data_path)
        x_train = x_train.astype(np.float32)
        print('Data Imported')

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
                outputs = model(data)
                loss = criteria(outputs, data)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()*data.size(0)

            train_loss = train_loss/x_train.shape[0]
            loss_array.append(train_loss)

            torch.save(model, checkpoint_path + '/DCAE.pth')

        return loss_array

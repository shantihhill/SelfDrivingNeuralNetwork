import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join, abspath
import time
import os
import sys
import argparse
import logging
import random
import pickle

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)



class WorldTransformer:
    def __init__(self, args, pretrained = True):
        super(WorldTransformer, self).__init__()

        # Set default values for all arguments
        self.path = getattr(args, 'path', '/data/abiomed_tmp/processed')
        self.seq_dim = getattr(args, 'seq_dim', 6)
        self.output_dim = getattr(args, 'output_dim', 11*6)
        self.bc = getattr(args, 'bc', 64)
        self.nepochs = getattr(args, 'nepochs', 20)
        self.encs = getattr(args, 'encs', 2)
        self.lr = getattr(args, 'lr', 0.001)
        self.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)
        self.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
        self.dim_model = getattr(args, 'dim_model', 256)
        self.args = args

        self.device = 'mps'

        self.model = TimeSeriesTransformer(input_dim=self.seq_dim, output_dim=self.output_dim, dim_model=self.dim_model,
                                                num_encoder_layers = self.encs, pl_shape = 20,
                                                encoder_dropout = self.encoder_dropout, 
                                                decoder_dropout = self.decoder_dropout, 
                                                device=self.device)
        # self.train_loader = self.read_data('train')
        # self.test_loader = self.read_data(mode = 'test') 
        
        if args.data_name == 'train':   
            self.train_loader = self.read_data('train')
        else:
            self.test_loader = self.read_data('test')
        if pretrained:
            self.trained_model = self.load_model()
            print('loaded model')
        else:
            self.trained_model = self.train_model()
            print('trained model')
        self.rwd_mean = None
        self.rwd_std = None

    def train_model(self):

        start_time = time.time()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        train_losses = []
        print(f"number of epochs {self.nepochs}")
        for epoch in range(self.nepochs):
            self.model.train()
            total_loss = 0
            for src, pl, tgt in tqdm(self.train_loader):
                src = src.to(self.device)
                pl = pl.to(self.device)
                optimizer.zero_grad()
                output = self.model(src, pl)
                loss = criterion(output[:,0,:], tgt.reshape(10,50,66)[:,0,:].to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
            train_losses.append(avg_loss)

        
        if not os.path.exists('world_model/'):
            os.makedirs('world_model/')
        torch.save(self.model.to('cpu').state_dict(),  os.path.join('world_model', f"transformer_epoch_{self.nepochs}.pth"))
        print("World model total time: {:.3f}s".format(time.time() - start_time))
        return self.model


    def read_data(self, mode='train'):

        # if mode == 'train':
        dta = np.load('../cse-251-b-2025/train.npz')['data'].astype(np.float32)[:1000]
        
        self.rwd_mean = dta.mean(axis=(0, 1))
        self.rwd_std = dta.std(axis=(0, 1))
        horizon = 10
        if mode == 'test':
            dta = np.load('../cse-251-b-2025/test.npz')['data'].astype(np.float32)[:1000]
            horizon = 60

        x_n = (dta - self.rwd_mean) / self.rwd_std
        x, pl, y = self.prep_transformer_world(x_n, ts = horizon+1, dims = 6)
        print("plshape is ", pl.shape)
        # pl = pl[..., :horizon, :].reshape(x.shape[0], horizon*2)
        dataset = TimeSeriesDataset(x, pl, y)
        loader = DataLoader(dataset, batch_size=self.bc, shuffle=True)
        return loader

    def resize(self, obs, action, next_state):
        # resize [N,1080] to [N,90, 12]
        n = int(obs.shape[0]/(self.seq_dim*90))
        #dont take the p-level into the observation space
        x = obs.reshape(n,90,self.seq_dim)
        y = next_state.reshape((n, 60*self.seq_dim))
        action = action.reshape(-1,90)
        loader = DataLoader(TimeSeriesDataset(x, action, y), batch_size=self.bc, shuffle=False)
        return loader
    
    def load_model(self):
        # Load the model state dict
        self.model.load_state_dict(torch.load(os.path.join('world_model', f"transformer_epoch_{self.nepochs}.pth")))
        return self.model.to(self.device)
    
    def predict(self, obs_loader):

        batch = 0
        with torch.no_grad():
            all_outputs = []
            self.trained_model.eval()
            for src, pl, tgt in obs_loader: #why loader size 1
                outputs = []
                input_i = src
                for i in range(6):
                    pl_i = pl[:, i*10:(i+1)*10].to(self.device)
                    output = self.trained_model(input_i, pl_i)
                    output_reshaped = output.reshape([output.shape[0], 50, 11, self.seq_dim])[:,:, 1:,:] #only take new predictions, ignore first datapoint
                    outputs.append(output_reshaped)
                    input_i = torch.concat([input_i[:,:,10:,:].to(self.device), output_reshaped], axis=1)
                #64x90x6
                pred = np.array(torch.concat(outputs, axis=1).detach().cpu())
                # all_outputs.append(pred.detach().cpu())

        # final = np.array(all_outputs).reshape(-1,90,12)

        #calculate the MSE loss and print
        return pred  

    def prep_transformer_world(self, x_n, ts, dims=6):
            
            n = x_n.shape[0]
            x = x_n[:,:, :50, :].reshape((n, 50, 50, dims))
            y = x_n[:,:, 49:49+ts, :].reshape((n, 50*ts*dims))
            pl = x_n[:, :, 50:50+ts-1, :2].reshape((n,50, (ts-1)*2))
            return x, pl, y
    



class Decoder(nn.Module):
    def __init__(self, input_size=82, hidden_size1=512, hidden_size2=256, output_size=66, dropout=0):
        #output of the encoder+controlled variable (p-level)
        super(Decoder, self).__init__()
        # print(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
      


    def forward(self, x):
        # print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TimeSeriesDataset(Dataset):
    def __init__(self, data, pl, labels):
        self.data = data
        self.pl = pl
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.pl[idx], self.labels[idx]
    

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dim_model=512, num_heads=8, num_encoder_layers=3, encoder_dropout=0.1, 
                        decoder_dropout=0.1, max_len = 100, pl_shape=10, device='cpu'):
        super(TimeSeriesTransformer, self).__init__()
        
        self.device = torch.device(device)

        self.dim_model = dim_model
        self.input_embedding = nn.Linear(input_dim, dim_model).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dropout=encoder_dropout).to(self.device),
            num_layers=num_encoder_layers
        ).to(self.device)
        self.dim_model = dim_model
        
        self.positional_encoding = self.create_positional_encoding(max_len, dim_model).to(self.device)

        self.decoder = Decoder(input_size = dim_model+pl_shape, output_size = output_dim, dropout = decoder_dropout).to(self.device)
    
    def create_positional_encoding(self, max_len, dim_model):
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src, pl):
        src = self.input_embedding(src.to(self.device)) * np.sqrt(self.dim_model)
        #src = self.input_projection(src) * np.sqrt(self.dim_model)

        src += self.positional_encoding[:, :src.size(2)].clone().detach()
        
        src = src.reshape(src.shape[0], src.shape[1]*src.shape[2], -1)
        src = src.permute(1, 0, 2)
          # Transformer expects (seq_len, batch, features)
        encoded_src = self.transformer_encoder(src)
        encoded_src = encoded_src.permute(1, 0, 2)  # Back to (batch, seq_len, features)
        encoded_src = encoded_src.reshape(encoded_src.shape[0], 50, 50, -1)
        pp = torch.cat([encoded_src[:, :, -1, :], pl], 2)
        # print(encoded_src.shape)
        # print("pl shape forward", pl.shape)
        # if pp.shape[1] == 128:
        #     print(pp)
        output = self.decoder(pp)  # Use only the last time step
        #dimension of the dec input last dim of encoder
        return output
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=bool, default=False)

    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=0,
                    help="Which GPU device index to use"
                )


    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--device', type=str, default = torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument('-data_name', '--data_name', type=str, metavar='<size>', default='train',
                help='which data to work on.')
    #world transformer arguments
    parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=6,
                        help='Specify the sequence dimension.')
    parser.add_argument('-output_dim', '--output_dim', type=int, metavar='<dim>', default=11*6,
                        help='Specify the sequence dimension.')
    parser.add_argument('-bc', '--bc', type=int, metavar='<size>', default=64,
                        help='Specify the batch size.') 
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=20, #change
                        help='Specify the number of epochs to train for.')
    parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=2,
                help='Set the number of encoder layers.') 
    parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.001,
                        help='Specify the learning rate.')
    parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0.1,
                help='Set the tunable dropout.')
    parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256,
                help='Set the number of encoder layers.')
    parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
                        default='/data/abiomed_tmp/processed',
                        help='Specify the path to read data.')
    

    args = parser.parse_args()
    


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



    world_transformer = WorldTransformer(args, pretrained = False)


    loader = world_transformer.read_data(mode='train')




import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join, abspath
import time
import os
import sys
import math
import argparse
import logging
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]
        # Getting 50 historical timestamps and 60 future timestamps
        hist = scene[:, :50, :].copy()    # (agents=50, time_seq=50, 6)
        future = torch.tensor(scene[0, 50:, :].copy(), dtype=torch.float32)  # (60, 2)
        
        # Data augmentation(only for training)
        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                # Rotate the historical trajectory and future trajectory
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future[..., :2] = future[..., :2] @ R
                future[..., 2:4] = future[..., 2:4] @ R

            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        # Use the last timeframe of the historical trajectory as the origin
        origin = hist[0, 49, :2].copy()  # (2,)
        hist[..., :2] = hist[..., :2] - origin

        future[..., :2] = future[..., :2] - origin

        # Normalize the historical trajectory and future trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        future[..., :4] = future[..., :4] / self.scale
         
        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

        return data_item
    

class TrajectoryDatasetTest(Dataset):
    def __init__(self, data, scale=10.0):
        """
        data: Shape (N, 50, 110, 6) Testing data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        """
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Testing data only contains historical trajectory
        scene = self.data[idx]  # (50, 50, 6)
        hist = scene.copy()
        
        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale
        # hist = hist[...,:-1]
        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
        return data_item    



class Decoder(nn.Module):
    def __init__(self, input_size=82, hidden_size1=256, hidden_size2=512, output_size=66, dropout=0):

        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # self.bn1 = nn.BatchNorm1d(hidden_size1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # self.bn2 = nn.BatchNorm1d(hidden_size2)

        self.fc3 = nn.Linear(hidden_size2, output_size)


    def forward(self, x):

        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dim_model=512, num_heads=8, num_encoder_layers=3, encoder_dropout=0.1, 
                        decoder_dropout=0.1, max_len = 100, device='cpu'):
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

        self.decoder = Decoder(input_size = dim_model, output_size = output_dim, dropout = decoder_dropout).to(self.device)
    
    def create_positional_encoding(self, max_len, dim_model):
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, src):
        
        src = self.input_embedding(src.to(self.device)) * np.sqrt(self.dim_model)
        #src = self.input_projection(src) * np.sqrt(self.dim_model)

        src += self.positional_encoding[:, :src.size(1)].clone().detach()
        
        # src = src.reshape(src.shape[0], src.shape[1]*src.shape[2], -1)
        src = src.permute(1, 0, 2) # Transformer expects (seq_len, batch, features)
        encoded_src = self.transformer_encoder(src)
        encoded_src = encoded_src.permute(1, 0, 2)  # Back to (batch, seq_len, features)
        # encoded_src = encoded_src.reshape(encoded_src.shape[0], 50, 50, -1)
        output = self.decoder(encoded_src[:, -1, :])  # Use only the last time step
        return output



class WorldTransformer:
    def __init__(self, args, pretrained = True):
        super(WorldTransformer, self).__init__()

        # Set default values for all arguments
        self.path = getattr(args, 'path', '')
        self.seq_dim = getattr(args, 'seq_dim', 5)
        self.output_dim = getattr(args, 'output_dim', 60*2)
        self.bc = getattr(args, 'bc', 64)
        self.nepochs = getattr(args, 'nepochs', 20)
        self.encs = getattr(args, 'encs', 2)
        self.lr = getattr(args, 'lr', 0.001)
        self.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)
        self.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
        self.dim_model = getattr(args, 'dim_model', 256)
        self.patience = getattr(args, 'patience', 15)
        self.weight_decay = getattr(args, 'weight_decay', 0.0001)
        self.args = args

        self.device = args.device

        self.model = TimeSeriesTransformer(input_dim=self.seq_dim, output_dim=self.output_dim, dim_model=self.dim_model,
                                                num_encoder_layers = self.encs,
                                                encoder_dropout = self.encoder_dropout, 
                                                decoder_dropout = self.decoder_dropout, 
                                                device=self.device)
        
        self.read_data()

        if pretrained:
            self.load_model()
            print('loaded model')
            
        else:
            
            self.train_model()
            print('trained model')

        
    def evaluate_model(self):
        self.model.eval()
        total_loss = 0
        total_loss_norm = 0
        criterion = nn.MSELoss()

        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()  
        sample_input = None
        sample_pred = None
        sample_target = None
        random_indices = random.sample(range(len(self.val_loader)), 4)
        i = 0
        with torch.no_grad():
            val_mae = 0
            for batch_idx, batch in enumerate(self.val_loader):
                batch = batch.to(self.device)
                batch_x = batch.x
                batch_x = batch_x.reshape(-1, 50, 50, args.seq_dim)[:, 0, :, :]
                context = batch_x.clone()            # (N, L_ctx, 2)
                y = batch.y.view(batch.num_graphs, 60, args.seq_dim)
                # 2) container for this batch’s 60‐step forecasts
                all_steps = []

                n_iters = math.ceil(60 / (args.output_dim/args.seq_dim))
                for _ in range(n_iters):
                    out_norm = self.model(context)               # (N, 11*2)
                    out_norm = out_norm.view(-1, 10, args.seq_dim)     # (N, 11, 2)

                    new10 = out_norm[:, :, :]              # (N, 10, 2)
                    all_steps.append(new10)

                
                    context = torch.cat([context.reshape(-1, 50, args.seq_dim)[:, 10:, :], out_norm[:, :, :]], dim=1)
                    
                # 3) stitch the 6 × 10 = 60 steps together
                pred_norm = torch.cat(all_steps, dim=1)      # (N, 60, 2)


                pred_unnorm = pred_norm[..., :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                y_unnorm = torch.stack(torch.split(batch.y[..., :2], 60, dim=0), dim=0) * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                # 4) de‐normalize
                # y_unnorm = y[..., :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                total_loss_norm += criterion(pred_norm, y).item() #norm mse
                val_mae += nn.L1Loss()(pred_unnorm, y_unnorm).item() #unnorm mae
                total_loss += criterion(pred_unnorm, y_unnorm).item() #unnorm mse


                if batch_idx in random_indices:
                    plot_trajectory(axes[i], pred_unnorm.cpu(), y_unnorm.cpu(), title=f"Sample {batch_idx}")
                    i += 1

        plt.savefig('figures/trajectory_encoder.png')
        avg_loss_norm = total_loss_norm / len(self.val_loader)
        avg_val_mae = val_mae / len(self.val_loader)
        # tqdm.write('unnorm val loss', avg_loss_norm)
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss_norm, avg_loss, avg_val_mae
    
    def train_model(self):

        start_time = time.time()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        
        # Exponential decay scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        early_stopping_patience = self.patience
        best_val_loss = float('inf')
        no_improvement = 0
        
        # Save initial state for comparison
        initial_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}

        criterion = nn.MSELoss()
        train_losses = []
        train_mae_losses = []
        val_losses = []
        val_mae_losses = []   
        print(f"number of epochs {self.nepochs}")
        for epoch in tqdm(range(self.nepochs), desc="Epoch", unit="epoch"):
            self.model.train()
            total_loss = 0
            mae_loss = 0
            total_mse = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                batch_x = batch.x
                batch_x = batch_x.reshape(-1, 50, 50, args.seq_dim)[:, 0, :, :]
                pred = self.model(batch_x).reshape(-1, int(args.output_dim/args.seq_dim),args.seq_dim)[..., :, :]


                y = batch.y.view(batch.num_graphs, 60, args.seq_dim)[:, :int(args.output_dim/args.seq_dim), :]
                optimizer.zero_grad()
                loss = criterion(pred, y)
                total_loss += loss #loss for 6 features

                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                pred_unnorm = pred[..., :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                y_unnorm = y[..., :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                train_mse = criterion(pred_unnorm, y_unnorm).item() #
                total_mse += train_mse

                mae_loss += nn.L1Loss()(pred_unnorm, y_unnorm) #mae loss for x and y
                
                
                # loss_val = criterion(output[:,0,:], tgt.reshape(-1,50,66)[:,0,:].to(self.device))
            
            scheduler.step()
            eval_loss_norm, eval_loss_unnorm, eval_loss_mae = self.evaluate_model()
            val_losses.append(eval_loss_unnorm)
            val_mae_losses.append(eval_loss_mae)
            tqdm.write(f'Val MSE in epoch {epoch} is {eval_loss_unnorm:.4f}',)
            tqdm.write(f'Val MAE in epoch {epoch} is {eval_loss_mae:.4f}',)

            avg_loss = total_mse / len(self.train_loader)
            train_losses.append(avg_loss)
            train_mae_losses.append(mae_loss.item()/ len(self.train_loader))
            tqdm.write(f'Epoch {epoch}, train MSE: {avg_loss:.4f}')
            tqdm.write(f'Epoch {epoch}, train MAE: {mae_loss.item()/ len(self.train_loader):.4f}')
            
            all_loss = [train_mae_losses, val_mae_losses]    

            if epoch % 5 == 0:
                # tqdm.write(f"Sample pred first 3 steps: {sample_pred[:3]}")
                # tqdm.write(f"Sample target first 3 steps: {sample_target[:3]}")
                
                # Check if model weights are changing
                if epoch > 0:
                    weight_change = False
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            initial_param = initial_state_dict[name]
                            if not torch.allclose(param, initial_param, rtol=1e-4):
                                weight_change = True
                                break
                    if not weight_change:
                        tqdm.write("WARNING: Model weights barely changing!")
            
            # Relaxed improvement criterion - consider any improvement
            if eval_loss_unnorm < best_val_loss:
                tqdm.write(f"Validation improved: {best_val_loss:.6f} -> {eval_loss_unnorm:.6f}")
                best_val_loss = eval_loss_unnorm
                no_improvement = 0
                torch.save(self.model.to('cpu').state_dict(),  os.path.join('world_model', f"transformer_epoch_{self.nepochs}.pth"))
                self.model.to(self.device)
            else:
                no_improvement += 1
                if no_improvement >= early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs without improvement")
                    break


        # plot_loss_curves(all_loss)
        plot_loss_curves([val_mae_losses])
        
        # if not os.path.exists('world_model/'):
        #     os.makedirs('world_model/')
        
        print("World model total time: {:.3f}s".format(time.time() - start_time))
        # return self.model


    def read_data(self):
        train_file = np.load('../cse-251-b-2025/train.npz')

        train_data = train_file['data'][...,:-1]
        print("train_data's shape", train_data.shape)
        
        torch.manual_seed(251)
        np.random.seed(42)
        
        scale = 7.0
        
        N = len(train_data)
        val_size = int(0.1 * N)
        train_size = N - val_size
        
        train_dataset = TrajectoryDatasetTrain(train_data[:train_size], scale=scale, augment=True)
        val_dataset = TrajectoryDatasetTrain(train_data[train_size:], scale=scale, augment=False)
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

    
    def load_model(self):
        # Load the model state dict
        self.model.load_state_dict(torch.load(os.path.join('world_model', f"transformer_epoch_{self.nepochs}.pth"), map_location=self.device))

    

    def prep_transformer_world(self, x_n, ts, dims=6):
            
            n = x_n.shape[0]
            x = x_n[:,:, :50, :].reshape((n, 50, 50, dims))
            y = x_n[:,:, 49:49+ts, :].reshape((n, 50*ts*dims))
            pl = x_n[:, :, 50:50+ts-1, :2].reshape((n,50, (ts-1)*2))
            return x, pl, y
    

    def plot_traj(self):
        # randomly select 4 samples from the validation set
        random_indices = random.sample(range(len(self.val_loader)), 4)
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()  # Flatten the array to iterate single axes objects
        self.model.eval()
        for i, idx in enumerate(random_indices):
            batch = self.val_loader[idx]
            batch = batch.to(self.device)
            pred = self.model(batch)
            gt = torch.stack(torch.split(batch.y, 60, dim=0), dim=0)

            pred = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            gt = torch.stack(torch.split(batch.y, 60, dim=0), dim=0) * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)

            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

            # Plot the trajectory using the i-th axis
            plot_trajectory(axes[i], pred, gt, title=f"Sample {idx}")

        


def pred_submission(scale = 7, device = 'mps'):
    
    test_file = np.load('../cse-251-b-2025/test_input.npz')
        
    test_data = test_file['data'][..., :-1]
    print("test_data's shape", test_data.shape)
    test_dataset = TrajectoryDatasetTest(test_data, scale=scale)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             collate_fn=lambda xs: Batch.from_data_list(xs))
    
    wm = WorldTransformer(args, pretrained = True)
    model = wm.model.to(device)
    model.eval()
    
    pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch_x = batch.x
            batch_x = batch_x.reshape(-1, 50, 50, args.seq_dim)[:, 0, :, :]
            context = batch_x.clone()            # (N, L_ctx, 2)

            # 2) container for this batch’s 60‐step forecasts
            all_steps = []

            #  we need 60 new points; each iteration gives us 10 real new ones
            n_iters = math.ceil(60 / (args.output_dim/args.seq_dim))
            for _ in range(n_iters):
                # a) run the model: returns (N, 11*6)
                out_norm = model(context)               # (N, 11*2)
                out_norm = out_norm.view(-1, 10, args.seq_dim)     # (N, 11, 2)

                # b) ignore the first output (overlap), keep next 10
                new10 = out_norm[:, :, :]              # (N, 10, 2)
                all_steps.append(new10)

             
                context = torch.cat([context.reshape(-1, 50, args.seq_dim)[:, 10:, :], out_norm[:, :, :]], dim=1)
                # — if your model expects a fixed-length input, make sure context
                #   stays the same length (L_ctx).  Here we assume L_ctx == 11,
                #   so dropping 10 and appending 11 keeps L_ctx=12, but you can
                #   slice exactly as needed:
                # context = context[:, -L_ctx:, :]

            # 3) stitch the 6 × 10 = 60 steps together
            pred_norm = torch.cat(all_steps, dim=1)      # (N, 60, 2)

            # 4) de‐normalize
            pred = pred_norm[..., :2] * batch.scale.view(-1,1,1) \
                + batch.origin.view(-1,1,2)

            pred_list.append(pred.cpu().numpy())

    # final output
    pred_list = np.concatenate(pred_list, axis=0)  # (Total_N, 60, 2)
    pred_output = pred_list.reshape(-1, 2)
    output_df = pd.DataFrame(pred_output, columns=['x', 'y'])
    output_df.index.name = 'index'
    output_df.to_csv(f'submission/submission_encoder_{args.nepochs}.csv', index=True)


def plot_loss_curves(losses):

    plt.figure()
    plt.plot(range(len(losses[0])), losses[0], label='Validation Loss')
    
    if len(losses) > 1:
        plt.plot(range(len(losses[0])), losses[1], label='Training Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves')
    plt.savefig(f'figures/loss_encoder_{args.nepochs}')
    # plt.show()

def plot_trajectory(ax, pred, gt, title=None):
    ax.cla()
    # Plot the predicted future trajectory
    ax.plot(pred[0,:60,0], pred[0,:60,1], color='red', label='Predicted Future Trajectory')
    
    # Plot the ground truth future trajectory
    ax.plot(gt[0,:60,0], gt[0,:60,1], color='blue', label='Ground Truth Future Trajectory')
    
    # Optionally set axis limits, labels, and title.
    x_max = max(pred[0,:60, 0].max(), gt[0,:60, 0].max())
    x_min = min(pred[0, :60,0].min(), gt[0,:60, 0].min())
    y_max = max(pred[0, :60,1].max(), gt[0, :60,1].max())
    y_min = min(pred[0,:60, 1].min(), gt[0,:60, 1].min())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    
    if title:
        ax.set_title(title)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--device', type=str, default = torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument('-data_name', '--data_name', type=str, metavar='<size>', default='train',
                help='which data to work on.')
    #world transformer arguments
    parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=5,
                        help='Specify the sequence dimension.')
    parser.add_argument('-output_dim', '--output_dim', type=int, metavar='<dim>', default=10*5,
                        help='Specify the sequence dimension.')
    parser.add_argument('-bc', '--bc', type=int, metavar='<size>', default=32,
                        help='Specify the batch size.') 
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=2, #change
                        help='Specify the number of epochs to train for.')
    parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=5,
                help='Set the number of encoder layers.') 
    parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.001,
                        help='Specify the learning rate.')
    parser.add_argument('-weight_decay', '--weight_decay', type=float, metavar='<size>', default=0.0005,
                        help='Specify the weight decay.')
    parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.5,
                help='Set the tunable dropout.')
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0.5,
                help='Set the tunable dropout.')
    parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256,
                help='Set the number of encoder layers.')
    parser.add_argument('-patience', '--patience', type=int, default=15,
                help='Set the patience for early stopping.')
    parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
                        default='',
                        help='Specify the path to read data.')


    args = parser.parse_args()



    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    world_transformer = WorldTransformer(args, pretrained = True)
    # pred_submission()

    world_transformer.evaluate_model()

    
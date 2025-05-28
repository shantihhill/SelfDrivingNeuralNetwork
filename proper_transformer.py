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
import argparse
import logging
import random
import pickle
import math
from torch.nn.functional import linear, softmax

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

#ignore warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*")

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, dim_model=512, num_heads=8, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 encoder_dropout=0.1, decoder_dropout=0.1, max_len=110, device='cpu'):
        super(TimeSeriesTransformer, self).__init__()
        
        self.device = torch.device(device)
        self.dim_model = dim_model
        
        # Input embedding
        self.input_embedding = nn.Sequential(
                nn.Linear(input_dim, dim_model),
                nn.LayerNorm(dim_model)
            ).to(self.device)
        # self.input_embedding_decoder = nn.Linear(output_dim, dim_model).to(self.device)

        self.input_embedding_decoder = nn.Sequential(
                nn.Linear(output_dim, dim_model),
                nn.LayerNorm(dim_model)
            ).to(self.device)
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_len, dim_model).to(self.device)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, 
            nhead=num_heads, 
            dropout=encoder_dropout,
             activation="relu",
            batch_first=True

        ).to(self.device)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(dim_model),
        ).to(self.device)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model, 
            nhead=num_heads, 
            dropout=decoder_dropout,
            batch_first=True,
            norm_first=True  
        ).to(self.device)
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(dim_model),
        ).to(self.device)
        
        
        # Output projection
        self.output_projection = nn.Linear(dim_model, output_dim).to(self.device)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    
    def create_positional_encoding(self, max_len, d_model):
        # Create positional encoding as in the original transformer paper
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None):
        """
        src: Source sequence (batch_size, src_seq_len, input_dim)
        tgt: Target sequence for teacher forcing (batch_size, tgt_seq_len, input_dim)
              If None, only the encoder will be used
        """
        # Embed input and add positional encoding
        
        # src = src.permute(1,0,2)  # [batch_size, input_dim, src_len]
        src_embedded = self.input_embedding(src)
        src_len = src.shape[1]
        src_embedded = src_embedded + self.positional_encoding[:, :src_len, :]
        assert torch.isfinite(src_embedded).all(), 'src_embedded contains NaN'
        memory = self.transformer_encoder(src_embedded, mask = src_mask)
        for name, p in self.transformer_encoder.named_parameters():
            if not torch.isfinite(p).all():
                print(name, "has NaNs or Infs!")
        assert torch.isfinite(memory).all(), 'memory contains NaN'
        
        if tgt is None:
           
            pred_horizon = 60  # Define this in your model class
            decoder_input = src # shape: [batch_size, 1, features]
            predictions = []

            for i in range(pred_horizon):
                
                if i == 0:
                    current_input = decoder_input
                else:
                    decoder_input = decoder_input[:, :-1, :]
                      # Use the last time step as initial input
                    current_input = torch.cat([decoder_input, *predictions], dim=1)
                decoder_input_emb = self.input_embedding_decoder(current_input)
                
                # Add positional encoding (for entire decoder input so far)
                pos_enc = self.positional_encoding[:, src_len:src_len + decoder_input_emb.size(1), :]
                decoder_input_emb = decoder_input_emb + pos_enc

                decoder_output = self.transformer_decoder(
                    decoder_input_emb, memory,
                    tgt_mask=None,  
                    memory_mask=memory_mask
                )
                
                next_point = self.output_projection(decoder_output[:, -1:, :]).view(src.shape[0], 1, -1)
                predictions.append(next_point)
                
            # Concatenate all predictions
            return torch.cat(predictions, dim=1)  # [batch_size, pred_horizon, output_features]
        
        # Teacher forcing mode with provided targets
        else:
           
            # Embed target and add positional encoding
            tgt_embedded = self.input_embedding_decoder(tgt)
            tgt_len = tgt.shape[1]
            assert not torch.isnan(tgt_embedded).any(), 'tgt_embedded contains NaN'
            # Add positional encoding, continuing from the encoder sequence
            # Positions start from src_len for the first decoder token
            decoder_positions = torch.arange(src_len, src_len + tgt_len).unsqueeze(0).unsqueeze(-1)
            decoder_positions = decoder_positions.expand(tgt.shape[0], -1, 1).to(self.device)
            tgt_embedded = tgt_embedded + self.positional_encoding[:, src_len:src_len + tgt_len, :]
            assert not torch.isnan(tgt_embedded).any(), 'tgt_embedded 2 contains NaN'
            # Generate a causal mask for the decoder self-attention if not provided
            # if tgt_mask is None:
            #     tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(self.device)
            # assert not torch.isnan(tgt_mask).any(), 'tgt_mask contains NaN'
            # tgt_embedded = tgt_embedded.permute(1,0,2)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(self.device)

            output = self.transformer_decoder(
                tgt_embedded[:, :, :], memory, 
                tgt_mask=tgt_mask, 
                # memory_mask=memory_mask
            )
            
            for name, p in self.transformer_decoder.named_parameters():
                if not torch.isfinite(p).all():
                    print(name, "has NaNs or Infs!")
            assert not torch.isnan(output).any(), 'output contains NaN'
            # output =  output.permute(1, 0,2)
            # Project to output dimensions for trajectory coordinates
            output = self.output_projection(output)
          
            return output
        
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -1e7).masked_fill(mask == 1, float(0.0))
        return mask
    
def predict(model, test_loader, target_len, src_mask=None):
    """
    Autoregressive prediction for a given sequence
    src: Input sequence (batch_size, src_seq_len, input_dim)
    target_len: Number of time steps to predict
    """
    outputs = []
    pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.x.to(args.device)
            batch_size = batch.size(0)
            input_dim = batch.size(2)
            batch = batch.reshape(-1, 50, 50, args.seq_dim)[:, 0, :, :]
            # Encode the source sequence
            memory = model()
            
            # Initialize first decoder input (could be the last element of src or zeros)
            decoder_input = batch[:, -1:, :]  # Use the last time step as initial input
            
            predictions = []
            # Autoregressive generation
            for i in range(target_len):
                # Pass through the decoder
                output = model(
                    batch, 
                    tgt=decoder_input,
                    # src_mask=src_mask
                ).reshape(-1,target_len,2)
                
                # Extract the latest prediction
                latest_prediction = output[:, -1:, :]
                predictions.append(latest_prediction)
                
                # Update decoder input for next time step
                decoder_input = torch.cat([decoder_input, latest_prediction], dim=1)
            
            # Concatenate all outputs
            outputs = torch.cat(predictions, dim=1)
            pred_list.append(outputs)
            # Reshape the prediction to (N, 60, 2)
        pred_norm = torch.cat(pred_list, dim=0).reshape(-1, 2) 
        pred = pred_norm * batch.scale.view(-1,1,1) + batch.origin.unsqueeze(1)
        output_df = pd.DataFrame(pred, columns=['x', 'y'])
        output_df.index.name = 'index'
        output_df.to_csv(f'submission/submission_proper_transformer_{args.nepochs}.csv', index=True)

    return outputs



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
        hist = scene[:, :50, :5].copy()    # (agents=50, time_seq=50, 6)
        future = torch.tensor(scene[0, 50:, :5].copy(), dtype=torch.float32)  # (60, 2)
        
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
       

        # Create a Data object for PyTorch Geometric

         
        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

        return data_item

def evaluate_model(model, val_loader, args):
    model.eval()
    total_loss = 0
    total_loss_norm = 0
    val_mae = 0
    criterion = nn.MSELoss()

    sample_input = None
    sample_pred = None
    sample_target = None
    
    with torch.no_grad():
        val_mae = 0
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(args.device)
            batch_x = batch.x
            batch_x = batch_x.reshape(-1, 50, 50, 5)[:, 0, :, :2]
            context = batch_x.clone()            # (N, L_ctx, 2)
            y = batch.y.view(batch.num_graphs, 60, 5)[...,:2]
            # 2) container for this batch’s 60‐step forecasts
            # all_steps = []
            pred_norm = model(batch_x, tgt=None)
            # n_iters = math.ceil(60 / (args.output_dim/args.seq_dim))
            # for _ in range(n_iters):
            #     out_norm = model(context)               # (N, 11*2)
            #     out_norm = out_norm.view(-1, int(args.output_dim/args.seq_dim), args.seq_dim)     # (N, 11, 2)

            #     new10 = out_norm.reshape(batch_x.shape[0], 60, args.seq_dim)          # (N, 10, 2)
            #     all_steps.append(new10)
            #     context = torch.cat([context[:, int(args.output_dim/args.seq_dim):, :], out_norm[context.shape[0],int(args.output_dim/args.seq_dim) , :]], dim=1)
                
            # pred_norm = torch.cat(all_steps, dim=1)      # (N, 60, 2)
            pred_unnorm = pred_norm[..., :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            y_unnorm = torch.stack(torch.split(batch.y[..., :2], 60, dim=0), dim=0) * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            
            # y_unnorm = y[..., :2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            total_loss_norm += criterion(pred_norm, y).item() #norm mse
            val_mae += nn.L1Loss()(pred_unnorm, y_unnorm).item() #unnorm mae
            total_loss += criterion(pred_unnorm, y_unnorm).item() #unnorm mse

        avg_loss_norm = total_loss_norm / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        # tqdm.write('unnorm val loss', avg_loss_norm)
        avg_loss = total_loss / len(val_loader)
        return avg_loss_norm, avg_loss, avg_val_mae


def train_model(model, train_loader, val_loader, args):

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    # optimizer, 
    # max_lr=args.lr,
    # total_steps=args.nepochs * len(train_loader),
    # pct_start=0.001  # 10% warmup
    # )
    scheduler =torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    early_stopping_patience = args.patience
    best_val_loss = float('inf')
    no_improvement = 0
    
    # Save initial state for comparison
    initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    print(f"number of epochs {args.nepochs}")
    # torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(args.nepochs), desc="Epoch", unit="epoch"):
        model.train()
        total_loss = 0
        train_mse = 0
        for batch in train_loader:

            batch = batch.to(args.device)
            batch_x = batch.x
            assert torch.isfinite(batch_x).all(), 'batch_x contains NaN'
            
            batch_x = batch_x.reshape(-1, 50, 50, 5)[:, 0, :, :2]
            ground_truth = batch.y.reshape(-1, 60, 5)[:, :, :2] 
            start_token = batch_x[:, -1, :].reshape(-1,1,2)  
            tgt_input = ground_truth[:,  :-1, :]  
            decoder_input = torch.cat([start_token, tgt_input], dim=1)
            assert torch.isfinite(decoder_input).all(), 'decoder_input contains NaN'
            pred = model(batch_x, decoder_input)
            # pred = model(batch_x, decoder_input).reshape(-1,60,2)
            if pred.isnan().any():
                print(f"nan in pred {epoch}")
                break
            y = batch.y.view(batch.num_graphs, 60, 5)[:, :, :2] 

            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            # for name, p in model.named_parameters():
            #     if p.grad is not None and not torch.isfinite(p.grad).all():
            #         raise RuntimeError(f"Non-finite gradient in {name}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            # print(optimizer.param_group s[0]['lr'])
            
            # print(optimizer.param_groups[0]['lr'])
            pred_unnorm = pred[...,:2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            y_unnorm = y[...,:2] * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            
            train_mse += criterion(pred_unnorm, y_unnorm).item()
            
            total_loss += loss.item()
            
            # loss_val = criterion(output[:,0,:], tgt.reshape(-1,50,66)[:,0,:].to(self.device))
        
        
        eval_loss_norm, eval_loss, eval_loss_mae = evaluate_model(model, val_loader, args)
        # lr_scheduler_val.step(eval_loss_norm)
        val_losses.append(eval_loss) #unnorm
        tqdm.write(f'Val loss in epoch {epoch} is {eval_loss:.4f}',) #unnorm
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_mse = train_mse / len(train_loader) 
        train_losses.append(avg_mse) #unnorm
        tqdm.write(f'Epoch {epoch}, train Loss: {avg_mse:.4f}') #unnorm
        
        all_loss = [train_losses, val_losses]    

        if epoch % 5 == 0:
            # tqdm.write(f"Sample pred first 3 steps: {sample_pred[:3]}")
            # tqdm.write(f"Sample target first 3 steps: {sample_target[:3]}")
            
            # Check if model weights are changing
            if epoch > 0:
                weight_change = False
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        initial_param = initial_state_dict[name]
                        if not torch.allclose(param, initial_param, rtol=1e-4):
                            weight_change = True
                            break
                if not weight_change:
                    tqdm.write("WARNING: Model weights barely changing!")
        
        # Relaxed improvement criterion - consider any improvement
        if eval_loss_norm < best_val_loss:
            tqdm.write(f"Validation improved: {best_val_loss:.6f} -> {eval_loss_norm:.6f}")
            best_val_loss = eval_loss_norm
            no_improvement = 0
            torch.save(model.to('cpu').state_dict(),  os.path.join('world_model', f"transformer_epoch_{args.nepochs}.pth"))
            model.to(args.device)
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs without improvement")
                break


    plot_loss_curves(all_loss)
    
    print("World model total time: {:.3f}s".format(time.time() - start_time))
    # return self.model

def pred_submission(model, test_loader,device = 'mps'):
    
        
        model.eval()
        
        pred_list = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred_norm = model(batch).to(device).reshape(-1,60,2)
                
                # Reshape the prediction to (N, 60, 2)
                pred = pred_norm * batch.scale.view(-1,1,1) + batch.origin.unsqueeze(1)
                pred_list.append(pred.cpu().numpy())
        pred_list = np.concatenate(pred_list, axis=0)  # (N,60,2)
        pred_output = pred_list.reshape(-1, 2)  # (N*60, 2)
        output_df = pd.DataFrame(pred_output, columns=['x', 'y'])
        output_df.index.name = 'index'
        output_df.to_csv(f'submission_transformer_{args.nepochs}.csv', index=True)


def read_data():
    train_file = np.load('../cse-251-b-2025/train.npz')

    train_data = train_file['data'][..., :-1]
    print("train_data's shape", train_data.shape)
    
    torch.manual_seed(251)
    np.random.seed(42)
    
    scale = 10.0
    
    N = len(train_data)
    val_size = int(0.1 * N)
    train_size = N - val_size
    
    train_dataset = TrajectoryDatasetTrain(train_data[:train_size], scale=scale, augment=False)
    val_dataset = TrajectoryDatasetTrain(train_data[train_size:], scale=scale, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))
    return train_loader, val_loader


def plot_loss_curves(losses):

    plt.figure()
    plt.plot(range(len(losses[0])), losses[0], label='Training Loss')
    plt.plot(range(len(losses[0])), losses[1], label='Validation Loss')  
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves')
    plt.savefig(f'figures/loss_transformer_{args.nepochs}')
    # plt.show()


def load_model(model_s):
    # Load the model state dict
    return model_s.load_state_dict(torch.load(os.path.join('world_model', f"transformer_epoch_{args.nepochs}.pth"), map_location = 'mps'))


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

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
        return data_item  
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--device', type=str, default = torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument('-data_name', '--data_name', type=str, metavar='<size>', default='train',
                help='which data to work on.')
    
    parser.add_argument('-pretrained', '--pretrained', action='store_true', default=False,
                        help='Load pretrained model.')
    #world transformer arguments

    parser.add_argument('-seq_dim', '--seq_dim', type=int, metavar='<dim>', default=2,
                        help='Specify the sequence dimension.')
    parser.add_argument('-output_dim', '--output_dim', type=int, metavar='<dim>', default=2,
                        help='Specify the sequence dimension.')
    parser.add_argument('-bc', '--bc', type=int, metavar='<size>', default=32,
                        help='Specify the batch size.') 
    parser.add_argument('-nepochs', '--nepochs', type=int, metavar='<epochs>', default=2, #change
                        help='Specify the number of epochs to train for.')
    parser.add_argument('-encoder_size', '--encs', type=int, metavar='<size>', default=1,
                help='Set the number of encoder layers.') 
    parser.add_argument('-decoder_size', '--decs', type=int, metavar='<size>', default=1,
                help='Set the number of decoder layers.')
    parser.add_argument('-lr', '--lr', type=float, metavar='<size>', default=0.002,
                        help='Specify the learning rate.')
    parser.add_argument('-weight_decay', '--weight_decay', type=float, metavar='<size>', default=0.0001,
                        help='Specify the weight decay.')
    parser.add_argument('-encoder_dropout', '--encoder_dropout', type=float, metavar='<size>', default=0.3,
                help='Set the tunable dropout.')
    parser.add_argument('-decoder_dropout', '--decoder_dropout', type=float, metavar='<size>', default=0.3,
                help='Set the tunable dropout.')
    parser.add_argument('-dim_model', '--dim_model', type=int, metavar='<size>', default=256, #must be at least 256
                help='Set the number of encoder layers.')
    parser.add_argument('-patience', '--patience', type=int, default=20,
                help='Set the patience for early stopping.')
    parser.add_argument('-path', '--path', type=str, metavar='<cohort>', 
                        default='',
                        help='Specify the path to read data.')


    args = parser.parse_args()



    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
  

    model = TimeSeriesTransformer(input_dim=args.seq_dim, output_dim=args.output_dim, dim_model=args.dim_model,
                                                    num_encoder_layers = args.encs, num_decoder_layers = args.decs,
                                                    encoder_dropout = args.encoder_dropout, 
                                                    decoder_dropout = args.decoder_dropout, 
                                                    device=args.device)
            
            

    if args.pretrained:
        saved_model = load_model(model)
        print('loaded model')
        test_file = np.load('../cse-251-b-2025/test_input.npz')
            
        test_data = test_file['data']
        print("test_data's shape", test_data.shape)
        test_dataset = TrajectoryDatasetTest(test_data, scale=7)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                collate_fn=lambda xs: Batch.from_data_list(xs))
        # predict(model, test_loader, 60, src_mask=None)
        pred_submission(saved_model, test_loader)
        
    else:
        train_loader, val_loader = read_data()
        train_model(model, train_loader, val_loader, args)
        print('trained model')
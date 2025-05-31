best_model = 'encoder_mlp_velocity'
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import tqdm
import sys
import argparse




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
        future = torch.tensor(scene[0, 50:, :4].copy(), dtype=torch.float32)  # (50, 60, 2)
        #add the feature of the scene number for each sample
        # Data augmentation(only for training)
        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                # Rotate the historical trajectory and future trajectory
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R

                # future = future @ R
                future[..., 0:2] = future[..., 0:2] @ R

                future[..., 2:4] = future[..., 2:4] @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1
                future[..., 2] *= -1

        # Use the last timeframe of the historical trajectory as the origin
        origin = hist[0, 49, :2].copy()  # (2,)
        hist[..., :2] = hist[..., :2] - origin
        future[..., :2] = future[..., :2] - origin
        # future = future - origin

        # Normalize the historical trajectory and future trajectory
        hist[..., :4] = hist[..., :4] / self.scale
        future = future / self.scale
        # hist[..., :2] = hist[..., :2] / self.scale
        # future[..., :2] = future[..., :2] / self.scale

        
        # print("hist's shape", hist.shape)
        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            y=future.type(torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0), # (1,2)
            scale=torch.tensor(self.scale, dtype=torch.float32), # scalar e.g. 7.0
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
        # hist = hist[...,]
        
        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale
        hist[..., :2] = hist[..., :2] / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
        return data_item
    
"""class AgentEncoder(nn.Module):
    def __init__(self, input_dim, d_model, agent_type_name, max_len=50):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.agent_type_embedding = nn.Parameter(torch.randn(1, 1, d_model))  # Learnable
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_model))  # Learnable

    def forward(self, x, mask= None):
        B, N, T, F = x.shape
        x = self.input_proj(x)
        x = x + self.agent_type_embedding + self.positional_encoding[:, :T]
        x = x.view(B * N, T, -1)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        # print('enc done')
        return output.view(B, N, -1), mask"""
class AgentEncoder(nn.Module):
    def __init__(self, input_dim, d_model, max_len=50, n_heads=8, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, d_model))  # [1, T, D]

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model),)

    def forward(self, x):
        """
        x: [B, N, T, F]
        mask: [B, N] — bool mask (True = valid agent, False = padded)
        """
        B, T, F = x.shape
        x = self.input_proj(x)  # [B, N, T, D]
        x = x + self.positional_encoding[:, :T]  # [B, N, T, D]

        # x = x.view(B * N, T, -1)  # [B*N, T, D]
        out = self.transformer_encoder(x)  # [B*N, T, D]
        # last_token = out[:, -1, :]         # [B*N, D]

        return out
    
class VehicleTrajectoryDecoder(nn.Module):
    def __init__(self, d_model=128, output_dim=2, n_heads=8, T_pred=60):
        super().__init__()
        self.T_pred = T_pred
        self.spatial_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.temporal_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.temporal_decoder = nn.TransformerDecoder(self.temporal_decoder_layer, num_layers=1)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, T_pred, d_model))

    def forward(self, H_v_all, H_v0_init):
        B, _, D = H_v_all.shape
        tgt = H_v0_init  # [B, 1, 1, D]
        outputs = []

        for t in range(self.T_pred):
            tgt_pe = tgt + self.positional_encoding[:, :t+1].unsqueeze(1)  # [B, 1, t+1, D]
            tgt_step = tgt_pe.view(B, t + 1, D)
            memory = H_v_all  # [B, N_v, D]
            # print(memory.shape, tgt_step[:, -1:, :].shape)

            # Use memory of all agents for spatial context
            attn_out, _ = self.spatial_attn(tgt_step[:, -1:, :], memory, memory)
            context = attn_out  # [B, 1, D]
            # print("context shape", context.shape)

            # Run temporal decoder
            decoded = self.temporal_decoder(tgt_step, memory)  # [B, t+1, D]
            next_token = decoded[:, -1:, :] + context  # Add spatial interaction

            tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=2)  # [B, 1, t+2, D]
            outputs.append(next_token.unsqueeze(1))  # [B, 1, 1, D]

        out_seq = torch.cat(outputs, dim=2)  # [B, 1, T_pred, D]
        coords = self.output_proj(out_seq)  # [B, 1, T_pred, output_dim]
        return coords
    
class VehicleTrajectoryPredictor(nn.Module):
    def __init__(self, input_dim=6, d_model=512, output_dim=2, n_heads=4, T_pred=60):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.vehicle_encoder = AgentEncoder(input_dim, d_model,)
        self.ped_encoder = AgentEncoder(input_dim, d_model, )
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.decoder = VehicleTrajectoryDecoder(d_model, output_dim, n_heads, T_pred)
        # Add multi-layer prediction head for better results
        self.fc1 = nn.Linear(d_model*2, d_model*2)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model*2, output_dim)

    def forward(self,data):
        # Encode vehicles and pedestrians
        x = data.x[..., :self.input_dim] 
        x = x.view(-1, 50, 50, self.input_dim)   # [B, N, T, F]
        B = x.shape[0]  # Batch size
        k = 1
        ego_traj = x[:, 0, :, :]  # (batch, 50, 5)
        # Process through LSTM
        H_v = self.vehicle_encoder(ego_traj)
        # Extract final hidden state
        ego_features = H_v[:, -1, :]
        
        # CLOSEST NEIGHBOR
        # ---- DISTANCES TO OTHER AGENTS ----
        ego_pos = x[:, 0, 49, :2].unsqueeze(1)  # (batch, 1, 2)
        agent_pos = x[:, :, 49, :2]  # (batch, 50, 2)
        dists = torch.norm(agent_pos - ego_pos, dim=-1)  # (batch, 50)
        dists[:, 0] = float('inf')  # mask out ego
        
        _, neighbor_ids = torch.topk(dists, k=k, dim=1, largest=False)  # (batch, 3)
        
        # ---- ENCODE NEIGHBORS ----
        neighbor_out_list = []

        for i in range(k):
            idx = neighbor_ids[:, i]  # (batch,)
            neighbor_trajs = torch.stack([x[b, idx[b]] for b in range(B)], dim=0)  # (batch, 50, 5)

            H_p = self.ped_encoder(neighbor_trajs)
        neighbor_out_list = H_p[:, -1, :].unsqueeze(1)  # (batch, 1, d_model)
         
        """# v_m  = (x[:,:,:,5]==0).any(axis=(0,2))
        # p_m  = (x[:,:,:,5]==1).any(axis=(0,2))
        # v_m = v_m.unsqueeze(0).unsqueeze(2).repeat(B,1,50)
        # p_m = p_m.unsqueeze(0).unsqueeze(2).repeat(B,1,50)
        # batch_x_v = x[v_m]  # [B, N_v, T, F]
        # batch_x_v.view(B,-1, 50, 6).shape
        # batch_x_p = x[p_m]  # [B, N_p, T, F]
        # vehicles = batch_x_v.view(B, -1, 50, 6)  # [B, N_v, T, F]
        # peds = batch_x_p.view(B, -1, 50, 6)  # [B, N_p, T, F]
        
        # N_v = vehicles.shape[1]
        # N_p = peds.shape[1]
        
        # v_mask = torch.ones(B, N_v).bool().to(vehicles.device)  # Mask for vehicles
        # p_mask = torch.ones(B, N_p).bool().to(peds.device)  # Mask for pedestrians


        # H_v, _ = self.vehicle_encoder(vehicles, v_mask)  # [B, N_v, D]
        # H_p, _ = self.ped_encoder(peds, p_m""ask)   """       # [B, N_p, D]

        B, N_v, D = H_v.shape
        N_p = H_p.shape[1]

        # Cross-attention: vehicles attend to pedestrians
        # H_v_flat = H_v.view(B * N_v, 1, D)
        # H_p_exp = H_p.unsqueeze(1).expand(B, N_v, N_p, D).contiguous().view(B * N_v, N_p, D)
        # p_mask_exp = ~p_mask.unsqueeze(1).expand(B, N_v, N_p).contiguous().view(B * N_v, N_p)

        # attn_out, _ = self.cross_attn(H_v_flat, H_p_exp, H_p_exp, key_padding_mask=p_mask_exp)
        # attn_out, _ = self.cross_attn(H_v_flat, H_p_exp, H_p_exp,)
        # H_v_cross = attn_out.view(B, N_v, D)
        # H_v_enhanced = H_v + H_v_cross  # [B, N_v, D]
        all_features = torch.cat([ego_features.unsqueeze(1), neighbor_out_list], dim=-1)  # (num_layers, batch, hidden_dim * 2)
        # q = ego_features.unsqueeze(1)  # (B, 1, D)
        # k = v = neighbor_out_list.unsqueeze(1)  # (B, 1, D)

        # context, _ = self.cross_attn(q, k, v)  # (B, 1, D)
        # pooled = context.squeeze(1)
        
        # Process through prediction head
        features = self.relu(self.fc1(all_features))
        features = self.dropout(features)
        out = self.fc2(features)
        # Initial decoder input for vehicle[0]
        # H_v0_init = H_v_enhanced[:, 0:1, :].unsqueeze(2)  # [B, 1, 1, D]
        # pred_traj = self.decoder(H_v_enhanced, H_v0_init)  # [B, 1, T_pred, output_dim]
        # out = pred_traj.squeeze(1)
        return  out.reshape(-1,60,2) # [B, T_pred, output_dim]

def train_improved_model(model, train_dataloader, val_dataloader, 
                         device, criterion=nn.MSELoss(), 
                         lr=0.001, epochs=100, patience=15):
    """
    Improved training function with better debugging and early stopping
    """
    # Initialize optimizer with smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Exponential decay scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    early_stopping_patience = patience
    best_val_loss = float('inf')
    no_improvement = 0
    
    # Save initial state for comparison
    initial_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    
    for epoch in tqdm.tqdm(range(epochs), desc="Epoch", unit="epoch"):
        # ---- Training ----
        model.train()
        train_loss = 0
        num_train_batches = 0
        forcing_ratio = max(0.0, 1.0 - epoch / 50)
        
        for batch in train_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            y_all = batch.y.view(batch.num_graphs, 60, 4)
            y = y_all[..., 2:4]
            # y = y_all
            # Check for NaN predictions
            if torch.isnan(pred).any():
                print(f"WARNING: NaN detected in predictions during training")
                continue
                
            loss = criterion(pred, y)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Invalid loss value: {loss.item()}")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # More conservative gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item()
            num_train_batches += 1
        
        # Skip epoch if no valid batches
        if num_train_batches == 0:
            print("WARNING: No valid training batches in this epoch")
            continue
            
        train_loss /= num_train_batches
        
        # ---- Validation ----
        model.eval()
        val_loss = 0
        val_mae = 0
        val_mse = 0
        xy_loss = 0
        num_val_batches = 0
        
        # Sample predictions for debugging
        sample_input = None
        sample_pred = None
        sample_target = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = batch.to(device)
                pred = model(batch)
                y_all = batch.y.view(batch.num_graphs, 60, 4)
                y = y_all[..., 2:4]
                # y = y_all
                # origin = batch.origin.unsqueeze(1)
                # Store sample for debugging
                if batch_idx == 0 and sample_input is None:
                    sample_input = batch.x[0].cpu().numpy()
                    sample_pred = pred[0].cpu().numpy()
                    sample_target = y[0].cpu().numpy()
                
                # Skip invalid predictions
                if torch.isnan(pred).any():
                    print(f"WARNING: NaN detected in predictions during validation")
                    continue
                    
                batch_loss = criterion(pred, y).item()
                val_loss += batch_loss
                xy_pred = convert_xy(pred).to(device)
                xy_loss += criterion(xy_pred, y_all[..., :2]).item()

                
                # Unnormalize for real-world metrics
                # batch.scale turns scale from 7.0 or (1,) shape i.e. scalar to (B,) shape
                # batch.origin turns origin from (1,2) shape to (B,2)
                
                # then .view(-1, 1, 1) turns scale from (B,) to (B, 1, 1)
                # then .unsqueeze(1) turns origin from (B, 2) to (B, 1, 2)
                # because pred and y have shapes (B, 60, 2) so these transformations make them compatible for the calculation
                
                pred_unnorm = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                y_unnorm = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
                
                val_mae += nn.L1Loss()(pred_unnorm, y_unnorm).item()
                val_mse += nn.MSELoss()(pred_unnorm, y_unnorm).item()
                
                num_val_batches += 1
        
        # Skip epoch if no valid validation batches
        if num_val_batches == 0:
            print("WARNING: No valid validation batches in this epoch")
            continue
            
        val_loss /= num_val_batches
        val_mae /= num_val_batches
        val_mse /= num_val_batches
        xy_loss /= num_val_batches
        
        # Update learning rate
        scheduler.step()
        
        # Print with more details
        tqdm.tqdm.write(
            f"Epoch {epoch:03d} | LR {optimizer.param_groups[0]['lr']:.6f} | "
            f"Train MSE {train_loss:.4f} | Val MSE {val_loss:.4f} | "
            f"Val MAE {val_mae:.4f} | Val MSE {val_mse:.4f} | "
            f"XY Val MSE {xy_loss:.4f}"
        )
        
        # Debug output - first 3 predictions vs targets
        if epoch % 5 == 0:
            tqdm.tqdm.write(f"Sample pred first 3 steps: {sample_pred[:3]}")
            tqdm.tqdm.write(f"Sample target first 3 steps: {sample_target[:3]}")
            
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
                    tqdm.tqdm.write("WARNING: Model weights barely changing!")
        
        # Relaxed improvement criterion - consider any improvement
        if val_loss < best_val_loss:
            tqdm.tqdm.write(f"Validation improved: {best_val_loss:.6f} -> {val_loss:.6f}")
            best_val_loss = val_loss
            no_improvement = 0
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join("models", best_model+str(epoch)+".pth"))
        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs without improvement")
                break

    torch.save(model.state_dict(), os.path.join("models", best_model+str(epoch)+".pth"))
    # Load best model before returning
    model.load_state_dict(torch.load(os.path.join("models", best_model+str(epoch)+".pth")))
    return model
# Example usage
def train_and_evaluate_model(device):
    # Create model
    model = VehicleTrajectoryPredictor(input_dim = 5, output_dim = 60*2 , d_model = 512)
    model = model.to(device)
    
    # Train with improved function
    train_improved_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        # lr = 0.007 => 8.946
        lr=0.001,  # do not change this
        patience=20,  # More patience
        epochs=200
    )
    
    # Evaluate
    model.eval()
    test_mse = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch = batch.to(device)
            pred = model(batch)
            y_all = batch.y.view(batch.num_graphs, 60, 4)
            y = y_all[..., 2:4]


            
            # Unnormalize
            pred = pred * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            y = y * batch.scale.view(-1, 1, 1) + batch.origin.unsqueeze(1)
            
            test_mse += nn.MSELoss()(pred, y).item()
    
    test_mse /= len(val_dataloader)
    print(f"Val MSE: {test_mse:.4f}")
    
    return model

def predict_model(device):
    test_dataset = TrajectoryDatasetTest(test_data, scale=scale)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            collate_fn=lambda xs: Batch.from_data_list(xs))

    best_model2 = torch.load(os.path.join("models", best_model+str(107)+".pth"))
    model = VehicleTrajectoryPredictor(input_dim = 5, output_dim = 60*2 , d_model = 512).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.25) # You can try different schedulers
    # criterion = nn.MSELoss()

    model.load_state_dict(best_model2)
    model.eval()

    pred_list = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_vel_norm = model(batch)
            
            # Reshape the prediction to (N, 60, 2)
            pred_vel = pred_vel_norm * batch.scale.view(-1,1,1)
            
            # Get origin in meters (position at t=49 for ego)
            # origin = batch.origin  # (B, 1, 2)
            origin = batch.origin.unsqueeze(1)  # Ensure shape is (B, 1, 2)
            
            # Integrate velocity to get position over 60 steps
            dt = 0.1  # seconds per step
            pred_pos = [origin]  # list of (B, 1, 2)
            
            for t in range(60):
                next_pos = pred_pos[-1] + pred_vel[:, t:t+1, :] * dt  # (B, 1, 2)
                pred_pos.append(next_pos)
            
            # Concatenate positions across time steps
            pred_xy = torch.cat(pred_pos[1:], dim=1)  # skip initial origin, get (B, 60, 2)

            pred_list.append(pred_xy.cpu().numpy())
            
    pred_list = np.concatenate(pred_list, axis=0)  # (N,60,2)
    pred_output = pred_list.reshape(-1, 2)  # (N*60, 2)
    output_df = pd.DataFrame(pred_output, columns=['x', 'y'])
    output_df.index.name = 'index'
    output_df.to_csv('submission_transformer_v2.csv', index=True)

def convert_xy(pred_vel, origin=None):
    dt = 0.1  # seconds per step
    if origin:
        pred_pos = [origin]  # list of (B, 1, 2)
        for t in range(60):
            next_pos = pred_pos[-1] + pred_vel[:, t:t+1, :] * dt  # (B, 1, 2)
            pred_pos.append(next_pos)
        
        # Concatenate positions across time steps
        pred_xy = torch.cat(pred_pos[1:], dim=1)  # skip initial origin, get (B, 60, 2)
    else:
        pred_pos = [0]
        for t in range(60):
            next_pos = pred_pos[-1] + pred_vel[:, t:t+1, :] * dt  # (B, 1, 2)
            pred_pos.append(next_pos)
        pred_xy = torch.cat(pred_pos[1:], dim=1)  # skip initial origin, get (B, 60, 2)
    return pred_xy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--d_model", type=str, default="cuda:0")
    args = parser.parse_args()


    torch.manual_seed(251)
    np.random.seed(42)

    train_file = np.load('train.npz')

    train_data = train_file['data']
    # train_data = train_data[::2]
    print("train_data's shape", train_data.shape)
    test_file = np.load('test_input.npz')

    test_data = test_file['data']
    print("test_data's shape", test_data.shape)
    scale = 10.0 #why not 10

    N = len(train_data)
    val_size = int(0.1 * N)
    train_size = N - val_size

    train_dataset = TrajectoryDatasetTrain(train_data[:train_size], scale=scale, augment=True)
    val_dataset = TrajectoryDatasetTrain(train_data[train_size:], scale=scale, augment=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: Batch.from_data_list(x))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: Batch.from_data_list(x))

    # Set device for training speedup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA GPU")
    else:
        device = torch.device('cpu')


    # model = train_and_evaluate_model(args.device)

    predict_model(args.device)
import numpy as np
import pandas as pd

# ===BEST===
# w2 = 8.0242 / (8.0242 + 8.2087)
# w21 = 8.2087/ (8.0242 + 8.2087)
#============

enc_dec = pd.read_csv("submission_lstm_simple_auto6.csv") 
enc_mlp2 = pd.read_csv("submission_lstm_simple_auto8 (3).csv") #closest agent
enc_mlp3 = pd.read_csv("submission_lstm_simple_auto8_axayangvel.csv")  
enc_mlp4 = pd.read_csv("submission_lstm_simple_auto8_axayangvel_and_angacc.csv")
enc_mlp5 = pd.read_csv("submission_lstm_simple_nonzero_agent_v1.csv")
enc_mlp6 = pd.read_csv("submission_lstm_simple_fastest_v3.csv")
enc_mlp7 = pd.read_csv("submission_lstm_simple_fastest_closest_v1.csv")
enc_mlp8 = pd.read_csv("submission_lstm_simple_fastest_closest_nonzero_v1.csv") #good model

enc_mlp9 = pd.read_csv("submission_lstm_simple_2fastest_v1.csv") #idle

# Use the 'index' column from one of them as the base
base_index = enc_mlp7['index']

# Remove index column for averaging
preds1 = enc_dec.drop(columns=['index'])
preds2 = enc_mlp2.drop(columns=['index'])
preds3 = enc_mlp3.drop(columns=['index'])
preds4 = enc_mlp4.drop(columns=['index'])
preds5 = enc_mlp5.drop(columns=['index'])
preds6 = enc_mlp6.drop(columns=['index'])
preds7 = enc_mlp7.drop(columns=['index'])
preds8 = enc_mlp8.drop(columns=['index'])
preds9 = enc_mlp9.drop(columns=['index'])

loss1 = 8.0242
loss2 = 8.2087
loss3 = 8.3146
loss4 = 8.0804
loss5 = 8.1407
loss6 = 8.7819
loss7 = 8.3382
loss8 = 8.3260 
loss9 = 8.6652 

# Inverse losses for weighting
inv_loss1 = 1.0 / loss1
inv_loss2 = 1.0 / loss2
inv_loss3 = 1.0/ loss3
inv_loss4 = 1.0/ loss4
inv_loss5 = 1.0/ loss5
inv_loss6 = 1.0/ loss6
inv_loss7 = 1.0/ loss7
inv_loss8 = 1.0/ loss8
inv_loss9= 1.0/ loss9

weight_sum = (
                inv_loss1 +
                inv_loss2 +
                inv_loss3 +
                inv_loss4 +
                inv_loss5+
                inv_loss6 +
                inv_loss7 +
                inv_loss8 +
                inv_loss9 
              )

# Weighted average
weighted_preds = (
                    preds1 * inv_loss1 +
                    preds2 * inv_loss2 +
                    preds3 * inv_loss3 +
                    preds4 * inv_loss4 +
                    preds5 * inv_loss5 +
                    preds6 * inv_loss6 +
                    preds7 * inv_loss7 +
                    preds8 * inv_loss8 +
                    preds9 * inv_loss9 
                   ) / weight_sum

# Ensure index is int
base_index = base_index.astype(int)

# Reattach the original index column
ens_df = pd.concat([base_index, weighted_preds], axis=1)


# Optional: sanity check
assert ens_df['index'].is_unique, "Index column has duplicates!"
assert len(ens_df) == len(base_index), "Length mismatch!"

# Save the correct format
ens_df.to_csv("submission_ensemble_9_mlp_v1.csv", index=False)

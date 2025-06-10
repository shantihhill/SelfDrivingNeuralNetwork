import numpy as np
import pandas as pd

# ===BEST===
# w2 = 8.0242 / (8.0242 + 8.2087)
# w21 = 8.2087/ (8.0242 + 8.2087)
#============

#change to 1/loss
# w2 = 8.9464  / (8.9464+  8.88198)
# w1 = 8.88198/ (8.9464 + 8.88198)

# w1 = 1/8.0242
# w2 = 1/8.2087
enc_dec = pd.read_csv("submission_lstm_simple_auto6.csv") 
enc_mlp2 = pd.read_csv("submission_lstm_simple_auto8 (3).csv") #closest agent
enc_mlp3 = pd.read_csv("submission_lstm_simple_auto8_axayangvel.csv")  
enc_mlp4 = pd.read_csv("submission_lstm_simple_auto8_axayangvel_and_angacc.csv")
enc_mlp5 = pd.read_csv("submission_lstm_simple_nonzero_agent_v1.csv")
enc_mlp6 = pd.read_csv("submission_lstm_simple_fastest_v3.csv")
enc_mlp7 = pd.read_csv("submission_lstm_simple_fastest_closest_v1.csv")
enc_mlp8 = pd.read_csv("submission_lstm_simple_fastest_closest_nonzero_v1.csv") #good model

enc_mlp9 = pd.read_csv("submission_lstm_simple_2fastest_v1.csv") #idle



# enc_mlp10 = pd.read_csv("submission_lstm_simple_nonzero_2agents_v1.csv") NO WAYYY



# enc_mlp8 = pd.read_csv("submission_lstm_simple_avg_dist_v1.csv")


# # ens1 = (enc_dec+enc_mlp)/2
# ens1 = enc_dec*w1 + enc_mlp*w2

# ens1.index = ens1['index']
# ens1= ens1.drop(columns ="index")
# ens1.reset_index()
# ens1.index = ens1.index.astype(int)
# ens1.to_csv('submission_ensemble_2_mlp_v3_ps.csv')




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
preds10 = enc_mlp10.drop(columns=['index'])





loss1 = 8.0242
loss2 = 8.2087
loss3 = 8.3146
loss4 = 8.0804
loss5 = 8.1407
loss6 = 8.7819
loss7 = 8.3382
# loss8 = 8.0413
loss8 = 8.3260 #changed from 0.3260
# loss9 = 8.2146
loss9 = 8.6652 #give higher weight
# loss9 = 8.0652 #new
loss10 = 8.2146




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
inv_loss10= 1.0/ loss10

weight_sum = (
                inv_loss1 +
                inv_loss2 +
                inv_loss3 +
                inv_loss4 +
                    inv_loss5+
                    inv_loss6 +
                    inv_loss7 +
                    inv_loss8 +
                    inv_loss9 +
                    inv_loss10
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
                         preds9 * inv_loss9 +
                        preds10 * inv_loss10 

                         ) / weight_sum

# Ensure index is int
base_index = base_index.astype(int)

# Reattach the original index column
ens_df = pd.concat([base_index, weighted_preds], axis=1)


# Optional: sanity check
assert ens_df['index'].is_unique, "Index column has duplicates!"
assert len(ens_df) == len(base_index), "Length mismatch!"

# Save the correct format
ens_df.to_csv("submission_ensemble_10_mlp_v1.csv", index=False)
# SelfDrivingNeuralNetwork
CSE251B Competition

https://www.kaggle.com/competitions/cse-251-b-2025/overview

https://www.kaggle.com/code/jerryfan1031/data-loading-and-submission-preperation/notebook

We implemented ensemble of LSTMs with different agent types and prediction objectives. 
```notebooks/``` has trials with LSTM and transformer. The following notebooks are included in the ensemble model. The validation losses are carried to ```src/ensemble.py``` which is the main ensembling script based on weighted arithmetic mean. 
- ```notebooks/lstm-vx-vy-social-feat-eng.ipynb```: LSTM predicting v_x and v_y with feature engineering of a_x and a_y.
- ```notebooks/lstm-vx-vy-social-feat-eng4.ipynb```: LSTM predicting v_x and v_y with feature engineering of a_x, a_y, angular velocity, angu/ar acceleration.
- ```notebooks/lstm-vx-vy-social-highestvel-closest.ipynb```: LSTM predicting v_x and v_y with ego, the fastest, and the closest agents.
- ```notebooks/lstm-vx-vy-social-highestvel.ipynb```: LSTM predicting v_x and v_y with ego, and the fastest agents. Also includes LSTM using ego and the 2 fastest agents.
- ```notebooks/lstm-vx-vy-social-nonzero-closest-type.ipynb```: LSTM predicting v_x and v_y with ego, the closest and the least non-zero-padde/ agent.
- ```notebooks/lstm-vx-vy-social-nonzero.ipynb```: LSTM predicting v_x and v_y with ego, and the least non-zero-padded agent.
- ```notebooks/lstm-vx-vy-social2.ipynb```: LSTM predicting v_x and v_y with ego only.
- ```notebooks/lstm.ipynb```: LSTM predicting x and y with ego only.


```models/``` has the ensembled model checkpoints.



 

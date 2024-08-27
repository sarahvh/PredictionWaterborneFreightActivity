#!/usr/bin/env python
# coding: utf-8

# # Temporal Convolutional Network
# 
# [Tutorial](https://forecastegy.com/posts/multiple-time-series-forecasting-with-convolutional-neural-networks-in-python/)
# 
# - `kernel_size`: this is the size of each filter used in the TCN convolution layers.
# - `dilations`: value of the dilation interval of the filters: how many time units they should skip when applying the transformation.
# - `input_size_multiplier`: first value optimized during the automatic search.
# - `encoder_hidden_size`: size of the encoded representation outputted by the TCN, which is also the number of filters applied.
# - `context_size`: after the TCN emits its outputs, they are transformed again to represent the overall context of the time series information.
# - `decoder_hidden_size`: the number of units in the hidden layers of the feedforward neural network that acts as the decoder.
# - `learning_rate`: the learning rate used by the optimizer.
# - `max_steps`: maximum number of times the neural network will update its weights during training

# In[2]:


import os
import sys
import pandas as pd
from pathlib import Path
import shutil
import neuralforecast
import optuna
import torch
import torch.nn
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from neuralforecast.auto import AutoTCN
from neuralforecast.models import TCN
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import HuberLoss
import lightning.pytorch as pl
import numpy as np
import csv
import pytorch_lightning as pl
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
os.environ['TQDM_DISABLE'] = '1'

import pickle
SEED = 1234

#Show all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[3]:


#Change directory to parent directory if necessary
if os.getcwd() == '/home/USACE_Modeling':
    None
else:
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

par = os.getcwd() #Parent Directory
par = Path(par)
sys.path.append(str(par))


# In[4]:


#List of data files
data_model_dict = {1:'UARK_WCS_AIS_Compiled_NewData_No_Aggregation.csv', 2:'UARK_WCS_AIS_Compiled_NewData_Mixed.csv', 3:'UARK_WCS_AIS_Compiled_NewData_Self-Propelled, Dry Cargo.csv',4:'UARK_WCS_AIS_Compiled_NewData_Self-Propelled, Tanker.csv',5:'UARK_WCS_AIS_Compiled_NewData_Tug_Tow.csv'}
if os.getcwd() == '/home/USACE_Modeling':
    data_iden = int(sys.argv[1]) #For HPC
else:
    data_iden=1
        
study_name = data_model_dict[data_iden].replace('.csv','')


# In[5]:


def dataset(data_model_dir = data_model_dict,  data_num=data_iden, par_dir=par):    

    #Show all columns in dataframe
    pd.set_option('display.max_columns', None)
    begin_testing = '2020Q1'
    end_testing = '2020Q4'
    
    batch_size = 128  # set this between 32 to 128
    #Read Main Data
    wcs_df = pd.read_csv(par_dir / 'Data' / 'UARK_WCS_AIS_Compiled_NewData.csv')
    cons_loc = pd.read_csv(par_dir / 'Data' / 'location_with_consistent_data_newdata.csv')
    wcs_df = pd.merge(wcs_df,cons_loc,how='left',on='sub_folder')
    port_terminal = wcs_df[['sub_folder', 'folder']].drop_duplicates()
    
    #Read Data
    file_loc = par_dir / 'Data' / data_model_dir[data_num]
    df = pd.read_csv(file_loc)
    #Drop columns that start with 'dwell
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith('dwell_'.lower())])
    
    df = df[df["quarter"] <= end_testing]
    df = pd.merge(df, port_terminal, on="sub_folder", how="left")
    df["ds"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
    # Rename some columns
    df.rename(columns={"sub_folder": "terminal", "QuarterOfTheYear": "quarter_of_year", "folder": "port"}, inplace=True)
    targets = [f"C_{i}" for i in range(1, 10)]
    ais_features = [col for col in df.columns if col.startswith("stop_count") or col.startswith("dwell_per_stop")]
    # Melt the DataFrame 'df' to a long format
    data = pd.melt(
        df,
        id_vars=[
            "terminal",
            "port",
            "ds",
            "quarter",
        ]
        + ais_features,
        value_vars=targets,
        var_name="commodity",
        value_name="y",
    )
    
    # Create a new column 'key' that combines the values in the 'port', 'terminal', and 'commodity' columns
    data["unique_id"] = data["port"].astype(str) + "|" + data["terminal"].astype(str) + "|" + data["commodity"].astype(str)
    #Removing outliers
    outlier_terminals = pd.read_csv(par_dir / 'Data' / 'outlier_terminals.csv')
    outlier_terminals_commodity = pd.read_csv(par_dir / 'Data' / 'outlier_terminals_commodity.csv')
    #Remove records from data where terminals are in outlier_terminals
    data = data[~data['terminal'].isin(outlier_terminals['terminal'])]
    #Remove records from data where key is in outlier_terminals_commodity
    data = data[~data['unique_id'].isin(outlier_terminals_commodity['unique_id'])]
    
    data["year"] = data["ds"].dt.year
    # Create four binary columns (Q1, Q2, Q3, Q4) based on the "ds" column
    data["Q1"] = (data["ds"].dt.quarter == 1).astype(int)
    data["Q2"] = (data["ds"].dt.quarter == 2).astype(int)
    data["Q3"] = (data["ds"].dt.quarter == 3).astype(int)
    data["Q4"] = (data["ds"].dt.quarter == 4).astype(int)
    
    fut_exog_features = ais_features + ["Q1", "Q2", "Q3", "Q4"]
    past_exog_features = fut_exog_features + ["y"]
    
    data["unique_id"] = data["port"].astype(str) + "|" + data["terminal"].astype(str) + "|" + data["commodity"].astype(str)
    data["year"] = data["ds"].dt.year
    
    #Drop columns terminal, port, quarter, commodity
    data.drop(columns=["terminal", "port", "commodity", "year"], inplace=True)
    
    train_df, test_df = data[data["quarter"] < begin_testing], data[data["quarter"] >= begin_testing]
    
    #Drop quarter column
    train_df.drop(columns=["quarter"], inplace=True)
    test_df.drop(columns=["quarter"], inplace=True)
    
    
    return dict(
        train_df=train_df,
        test_df=test_df,
        fut_exog_features=fut_exog_features,
        past_exog_features=past_exog_features,
    )
dataset_return = dataset()
train_df = dataset_return['train_df']
test_df = dataset_return['test_df']
fut_exog_features = dataset_return['fut_exog_features']
past_exog_features = dataset_return['past_exog_features']


# In[7]:


class MetricsCallback(pl.Callback, pl.Trainer):
    def __init__(self, save_path):
        super().__init__()
        self.metrics = []
        self.save_path = save_path  # Path to save the CSV file
        self.epoch = []
        self.a_trn_loss = []
        self.a_val_loss = []
        self.a_val_MAE = {}
        self.a_val_RMSE = {}


    def on_init_end(self, trainer):
        self.a_trn_loss = np.ones(trainer.max_epochs) * np.inf
        self.a_val_loss = np.ones(trainer.max_epochs) * np.inf

    def on_validation_end(self, trainer, pl_module):
        self.epoch = np.append(self.epoch, trainer.current_epoch)        
        self.a_trn_loss.append(trainer.callback_metrics["train_loss"].item())
        self.a_val_loss.append(trainer.callback_metrics["valid_loss"].item())

    def on_train_end(self, trainer, pl_module):
        self.save_metrics_to_csv()

    def save_metrics_to_csv(self):
        csv_save_path = self.save_path + '/metrics.csv'
        with open(csv_save_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epochs", "train_loss", "valid_loss"])
            for epoch in range(len(self.a_val_loss)):
                writer.writerow([epoch, self.a_trn_loss[epoch], self.a_val_loss[epoch]])



def define_model(trial):
    trial_n = trial.number
    folder_name = study_name +'_' + str(trial_n)
    if os.getcwd() == '/home/USACE_Modeling':
        folder_name = '/home/USACE_Modeling/Outputs/TCN_Outputs/' + folder_name
    else:
        folder_name = r'Outputs\TCN_Outputs\\' + folder_name
    #Create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    logger = MetricsCallback(save_path = folder_name)
    early_stopping = pl.callbacks.EarlyStopping(monitor='valid_loss', patience=30, verbose=False, mode='min', min_delta=0)

    trainer_kwargs = {
        'accelerator':"cpu",
        'enable_checkpointing': False,
        'logger' : False,
        'callbacks' : [logger, early_stopping],
        'num_sanity_val_steps' : 0,
        'enable_model_summary' : False,
        'enable_progress_bar' : False,
    }

    h = 4
    learning_rate = 0.000001
    max_steps = 10000000
    batch_size = 128
    random_seed = 1234
    scaler_type = 'standard'
    input_size = 4
    loss = HuberLoss()
    futr_exog_list = fut_exog_features
    hist_exog_list = past_exog_features


    kernel_size = trial.suggest_int("kernel_size", 2, 10)
    dilations = trial.suggest_categorical("dilations", [[1,2,4,8],[1,2,4,8,16],[1,2,4,8,16,32]])
    encoder_hidden_size = trial.suggest_categorical("encoder_hidden_size", [16, 32, 64, 128, 256, 512])
    context_size = trial.suggest_int("context_size", 1, 20)
    decoder_hidden_size = trial.suggest_categorical("decoder_hidden_size", [16, 32, 64, 128, 256, 512])
    decoder_layers = trial.suggest_int("decoder_layers", 1, 5)


    # Define the model
    models = [TCN(
        h=h,
        input_size = input_size,
        kernel_size=kernel_size,
        dilations=dilations,
        encoder_hidden_size=encoder_hidden_size,
        encoder_activation="ReLU",
        context_size=context_size,
        decoder_hidden_size=decoder_hidden_size,
        decoder_layers=decoder_layers,
        futr_exog_list=futr_exog_list,
        hist_exog_list=hist_exog_list,
        learning_rate=learning_rate,
        max_steps=max_steps,
        loss=loss,
        batch_size = batch_size,
        random_seed=random_seed,
        scaler_type = scaler_type,
        **trainer_kwargs,
    )]

    tcn = NeuralForecast(models = models, freq="QS")

    tcn.fit(train_df,
              val_size=4,
              verbose=False,
              )

    #Return validation loss
    #Load metrics csv
    metrics_df = pd.read_csv(folder_name + '/metrics.csv')
    #Get last validation loss
    val_loss = metrics_df['valid_loss'].iloc[-1]


    #Save the model in folder_name
    NeuralForecast.save(tcn, path=folder_name , save_dataset=True, overwrite=True)

    return val_loss


n_wait_value=500
n_trials_optuna = 2000
def pruning_callback(study, trial):
    n_wait = n_wait_value
    if len(study.trials) < n_wait:
        return False

    best_value = study.best_value
    check_values = [
        t.value 
        for t in study.trials[-n_wait:]
        if t.state == optuna.trial.TrialState.COMPLETE
        ]
    if all(v > best_value for v in check_values):
        study.stop()
        print('Study Pruned', flush=True)
    return False

#For Optuna
if os.getcwd() == '/home/USACE_Modeling':
    storage_path = "/home/USACE_Modeling/Outputs/TCN_Outputs/TCN_study.log" #For Slurm
else:
    storage_path = r"Outputs\TCN_Outputs"

storage_url = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path),)
study = optuna.create_study(study_name = study_name, storage = storage_url, load_if_exists=True, direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))

if __name__ == "__main__":
    study = optuna.load_study(study_name = study_name, storage=storage_url)
    study.optimize(define_model,callbacks=[pruning_callback, MaxTrialsCallback(n_trials_optuna, states=(TrialState.COMPLETE,))], n_trials=10, gc_after_trial=True)


    #Keep the best model
    directory_path = par / 'Outputs' / 'TCN_Outputs' #For HPC
    for trial in study.trials:
        if (trial.state == optuna.trial.TrialState.COMPLETE and trial.number != study.best_trial.number) or (trial.state == optuna.trial.TrialState.FAIL or trial.state == optuna.trial.TrialState.PRUNED):
            folder_path = Path(directory_path) / (study_name + '_' + str(trial.number))
            #Delete folder if it exists
            if folder_path.exists():
                shutil.rmtree(folder_path)

    print('Tuning Complete')    


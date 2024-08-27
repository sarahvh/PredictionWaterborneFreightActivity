#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import warnings
#Seeds for reproducibility
import random as rn
import numpy as np
import matplotlib.pyplot as plt
SEED = 1234
np.random.seed(SEED)
rn.seed(SEED)
import datetime

warnings.filterwarnings("ignore")
import pandas as pd
import csv
import lightning.pytorch as pl
pl.seed_everything(SEED, workers=True)
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from lightning.pytorch.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import MAE, SMAPE, MAPE, PoissonLoss, QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
import torch
torch.manual_seed(SEED)
from pathlib import Path
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import gc


# In[2]:


#Change directory to parent directory if necessary
if os.getcwd() == '/home/USACE_Modeling':
    None
else:
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

par = os.getcwd() #Parent Directory
par = Path(par)
sys.path.append(str(par))


# In[3]:


#List of data files
data_model_dict = {1:'UARK_WCS_AIS_Compiled_NewData_No_Aggregation.csv', 2:'UARK_WCS_AIS_Compiled_NewData_Mixed.csv', 3:'UARK_WCS_AIS_Compiled_NewData_Self-Propelled, Dry Cargo.csv',4:'UARK_WCS_AIS_Compiled_NewData_Self-Propelled, Tanker.csv',5:'UARK_WCS_AIS_Compiled_NewData_Tug_Tow.csv'}
if os.getcwd() == '/home/USACE_Modeling':
    data_iden = int(sys.argv[1]) #For HPC
else:
    data_iden=1
        
study_name = data_model_dict[data_iden].replace('.csv','')


# In[4]:


def dataset(data_model_dir = data_model_dict,  data_num=data_iden, par_dir=par):    

    #Show all columns in dataframe
    pd.set_option('display.max_columns', None)
    begin_testing = '2020Q1'
    end_testing = '2020Q4'
    
    batch_size = 32  # set this between 32 to 128
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
    # Split the values in the 'quarter' column by "Q" and convert them to integers
    temp = df["quarter"].str.split("Q", expand=True).astype(int)
    # Calculate the number of quarters from the start of the data and add it as a new column 'quarter_from_start'
    df["year"] = temp[0]
    df["time_idx"] = (temp[0] - temp[0].iloc[0]) * 4 + temp[1] - 1
    df.rename(columns={"sub_folder": "terminal", "QuarterOfTheYear": "quarter_of_year", "folder": "port"}, inplace=True)
    target = [col for col in df.columns if col.startswith('C_')]
    ais_features = [col for col in df.columns if col.startswith("stop_count") or col.startswith("dwell_per_stop")]
    # Melt the DataFrame 'df' to a long format
    data = pd.melt(
        df,
        id_vars=[
            "terminal",
            "port",
            "quarter_of_year",
            "quarter",
            "year",
            "time_idx",
        ]
        + ais_features,
        value_vars=target,
        var_name="commodity",
        value_name="volume",
    )
    # Create a new column 'key' that combines the values in the 'port', 'terminal', and 'commodity' columns
    data["key"] = data["port"].astype(str) + "|" + data["terminal"].astype(str) + "|" + data["commodity"].astype(str)
    
    outlier_terminals = pd.read_csv(par_dir / 'Data' / 'outlier_terminals.csv')
    outlier_terminals_commodity = pd.read_csv(par_dir / 'Data' / 'outlier_terminals_commodity.csv')
    
    #Remove records from data where terminal is in outlier_terminals
    data = data[~data['terminal'].isin(outlier_terminals['terminal'])]
    #Remove records from data where key is in outlier_terminals_commodity
    data = data[~data['key'].isin(outlier_terminals_commodity['key'])]
    
    #Drop port, terminal, and commodity columns
    data = data.drop(columns=["port", "terminal", "commodity"])
    #Set quarter of year as string
    data['quarter_of_year'] = data['quarter_of_year'].astype(str)
    #Split into train and test
    train_df = data[data['quarter'] < begin_testing]
    test_df = data[data['quarter'] >= begin_testing]
    #Drop quarter
    train_df = train_df.drop(columns=['quarter'])
    test_df = test_df.drop(columns=['quarter'])
    max_prediction_length = 4
    max_encoder_length = 4
    training_cutoff = train_df["time_idx"].max() - max_prediction_length
        
    # Create training dataset
    training_ret = TimeSeriesDataSet(
        train_df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="volume",
        group_ids=["key"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["key"],
        time_varying_known_categoricals=["quarter_of_year"],
        time_varying_known_reals=ais_features + ["time_idx"],
        time_varying_unknown_reals=["volume"],
        target_normalizer=GroupNormalizer(groups=["key"], transformation="relu"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
    validation_ret = TimeSeriesDataSet.from_dataset(training_ret, train_df, predict=True, stop_randomization=True)
    
    # create dataloaders for model
    train_dataloader_ret = training_ret.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader_ret = validation_ret.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    
    #Create dictionry to return
    return_dict = {'train_dataloader':train_dataloader_ret, 'val_dataloader':val_dataloader_ret, 'training':training_ret, 'test':test_df, 'train':train_df}
    return return_dict

#Get values from dataset
data_dict = dataset()
train_dataloader = data_dict['train_dataloader']
val_dataloader = data_dict['val_dataloader']
training = data_dict['training']
train_set = data_dict['train']
test_set = data_dict['test']

#Clear memory
try:
    del data_dict
    gc.collect()
    torch.cuda.empty_cache()
except:
    None


# In[ ]:


n_epochs = 1000
n_trials_optuna = 2000
devices_value = 1
num_nodes_value = 1
accelerator_value = 'cpu'
strategy_value = 'auto'
early_stopping_patience_value = 50
n_wait_value=500
learning_rate_value = 0.00001
reduce_on_plateau_patience_value = 100

#Convert above parameters to config_dict 
config_dict = {'n_epochs':n_epochs, 'n_trials_optuna':n_trials_optuna, 'devices_value':devices_value, 'num_nodes_value':num_nodes_value, 'accelerator_value':accelerator_value, 'strategy_value':strategy_value, 'early_stopping_patience_value':early_stopping_patience_value, 'n_wait_value':n_wait_value, 'learning_rate_value':learning_rate_value, 'reduce_on_plateau_patience_value':reduce_on_plateau_patience_value}

class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self, save_path):
        super().__init__()
        self.metrics = []
        self.save_path = save_path  # Path to save the CSV file
        self.epoch = []
        self.a_trn_loss = {}
        self.a_val_loss = {}
        self.a_val_MAE = {}
        self.a_val_RMSE = {}

    def on_init_end(self, trainer):
        self.a_trn_loss = np.ones(trainer.max_epochs) * np.inf
        self.a_val_loss = np.ones(trainer.max_epochs) * np.inf

    def on_validation_end(self, trainer, pl_module):
        
        self.epoch = np.append(self.epoch, trainer.current_epoch)            
        self.a_trn_loss[trainer.current_epoch] = trainer.callback_metrics["train_loss"].item()
        self.a_val_loss[trainer.current_epoch] = trainer.callback_metrics["val_loss"].item()
        self.a_val_MAE[trainer.current_epoch] = trainer.callback_metrics["val_MAE"].item()
        self.a_val_RMSE[trainer.current_epoch] = trainer.callback_metrics["val_RMSE"].item()
        #Call save_metrics_to_csv
        self.save_metrics_to_csv()
            
    # def on_train_end(self, trainer, pl_module):
    #     self.save_metrics_to_csv()
        
    def save_metrics_to_csv(self):
        with open(self.save_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # The header row will be added only if the file is empty
            if file.tell() == 0:
                writer.writerow(["epochs", "train_loss", "val_loss", "val_MAE", "val_RMSE"])
            epoch = self.epoch[-1]
            writer.writerow([epoch, self.a_trn_loss[epoch], self.a_val_loss[epoch], self.a_val_MAE[epoch], self.a_val_RMSE[epoch]])
            # for epoch in range(len(self.epoch)):
            #     writer.writerow([epoch, self.a_trn_loss[epoch], self.a_val_loss[epoch], self.a_val_MAE[epoch], self.a_val_RMSE[epoch]])
                #Empty dictionary

def define_model(trial , config=config_dict, train_dataloader=train_dataloader, val_dataloader=val_dataloader, training=training):
    par_dir = os.getcwd() #Parent Directory
    par_dir = Path(par_dir)
    
    file_name = study_name+'_'+str(trial.number)
    
    gradient_clip_val_value = trial.suggest_float("gradient_clip_val", 0.01, 1.0, step=0.01)
    hidden_size_value = trial.suggest_categorical("hidden_size", [8, 16, 32, 64, 128, 256])
    lstm_layers_value = trial.suggest_categorical("lstm_layers", [1, 2])
    attention_head_size_value = trial.suggest_categorical("attention_head_size", [1, 2, 3, 4])
    hidden_continuous_size_value_divider = trial.suggest_categorical("hidden_continuous_size_divider", [1, 2, 4, 8])
    hidden_continuous_size_value = int(hidden_size_value / hidden_continuous_size_value_divider)
    dropout_value = trial.suggest_float("dropout", 0, 0.3, step=0.01)
    
    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            print('Duplicate parameters; Trial Pruned', flush=True)
            raise optuna.exceptions.TrialPruned()
    
    tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=config['learning_rate_value'],
    hidden_size=hidden_size_value,
    lstm_layers=lstm_layers_value,
    attention_head_size=attention_head_size_value,
    dropout=dropout_value,
    hidden_continuous_size=hidden_continuous_size_value,
    output_size=7,  # there are 7 quantiles by default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    loss=QuantileLoss(),
    reduce_on_plateau_patience=config['reduce_on_plateau_patience_value'],
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience_value'], min_delta = 0, verbose=False, mode='min')
    
    checkpoint = ModelCheckpoint(monitor='val_loss', dirpath=par_dir / 'Outputs' / 'TFT_Outputs', filename=file_name, save_top_k=1, mode='min', verbose=False, save_last=False, save_weights_only=True)
    
    log_callback = MetricsCallback(save_path=par_dir / 'Outputs' / 'TFT_Outputs' / (file_name + '.csv'))
    
    #Create trainer
    trainer = pl.Trainer(
        max_epochs=config['n_epochs'],
        accelerator=config['accelerator_value'],
        devices=config['devices_value'],
        num_nodes=config['num_nodes_value'],
        enable_model_summary=False,
        gradient_clip_val=gradient_clip_val_value,
        callbacks=[early_stopping, checkpoint, log_callback],
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        strategy=config['strategy_value'],
    )
    
    #Print number of parameters
    # print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")
            
    trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
    )
      
    checkpoint_path = str(par_dir / 'Outputs' / 'TFT_Outputs' / (file_name + '.ckpt'))
    if os.path.exists(checkpoint_path):
        model = tft.load_from_checkpoint(checkpoint_path=checkpoint_path)
        score = trainer.test(model, val_dataloader, verbose=False)
        score = score[0]['test_RMSE']
    else:
        score = None
    # score = trainer.callback_metrics["val_RMSE"].item()
    
    #Clear memory
    try:
        del tft
        del trainer
        del early_stopping
        del checkpoint
        del log_callback
        del gradient_clip_val_value
        del hidden_size_value
        del lstm_layers_value
        del attention_head_size_value
        del hidden_continuous_size_value_divider
        del hidden_continuous_size_value
        del dropout_value
        gc.collect()
        torch.cuda.empty_cache()
    except:
        None
    
    return score
    
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
    storage_path = "/home/USACE_Modeling/Outputs/TFT_Outputs/TFT_study.log" #For Slurm
else:
    storage_path = r"Outputs\TFT_Outputs"

storage_url = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path),)
study = optuna.create_study(study_name = study_name, storage = storage_url, load_if_exists=True, direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))


if __name__ == "__main__":
    study = optuna.load_study(study_name = study_name, storage=storage_url)
    study.optimize(define_model,callbacks=[pruning_callback, MaxTrialsCallback(n_trials_optuna, states=(TrialState.COMPLETE,))], n_trials=3, gc_after_trial=True)
    
    #Keep the best model
    directory_path = par / 'Outputs' / 'TFT_Outputs' #For Local
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.number != study.best_trial.number:
            #Delete if file exists
            file_path = Path(directory_path) / (study_name + '_' + str(trial.number) + '.ckpt')
            if file_path.exists():
                file_path.unlink()
            csv_path = Path(directory_path) / (study_name + '_' + str(trial.number) + '.csv')
            if csv_path.exists():
                csv_path.unlink()

    print('Tuning Complete')


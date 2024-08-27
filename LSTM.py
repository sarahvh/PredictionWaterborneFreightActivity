#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
from pathlib import Path


# In[2]:


#Change directory to parent directory if necessary
if os.getcwd() == '/home/USACE_Modeling':
    None
else:
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


# In[3]:


#Seeds for reproducibility
import pandas as pd
import random as rn
import numpy as np
import matplotlib.pyplot as plt
SEED = 1234
np.random.seed(SEED)
rn.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Use CPU

import tensorflow as tf
rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.losses import Huber, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
import optuna
import optuna.storages
from optuna.storages import RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
# tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(physical_devices[0], 'GPU')
#SHow list of physical devices


# In[4]:


par_dir = os.getcwd() #Parent Directory
par_dir = Path(par_dir)
sys.path.append(str(par_dir))
from Functions import create_dataset


# In[5]:


#For display
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.options.display.max_columns = None
pd.set_option('display.max_rows', None)


# In[6]:


#Read Main Data
wcs_df = pd.read_csv(par_dir / 'Data' / 'UARK_WCS_AIS_Compiled_NewData.csv')
cons_loc = pd.read_csv(par_dir / 'Data' / 'location_with_consistent_data_newdata.csv')
wcs_df = pd.merge(wcs_df,cons_loc,how='left',on='sub_folder')


# In[7]:


#List of data files
data_model_dir = {1:'UARK_WCS_AIS_Compiled_NewData_No_Aggregation.csv', 2:'UARK_WCS_AIS_Compiled_NewData_Mixed.csv', 3:'UARK_WCS_AIS_Compiled_NewData_Self-Propelled, Dry Cargo.csv',4:'UARK_WCS_AIS_Compiled_NewData_Self-Propelled, Tanker.csv',5:'UARK_WCS_AIS_Compiled_NewData_Tug_Tow.csv'}


# In[8]:


if os.getcwd() == '/home/USACE_Modeling':
    data_num = int(sys.argv[1]) #For HPC
else:
    data_num=1
#Read Data
file_loc = par_dir / 'Data' / data_model_dir[data_num]
df = pd.read_csv(file_loc)
#Drop columns that start with 'dwell
df = df.drop(columns=[col for col in df.columns if col.lower().startswith('dwell_'.lower())])

#Removing outliers
outlier_terminal = pd.read_csv(par_dir / 'Data' / 'outlier_terminals.csv')
#Remove records from outlier terminals whose sub_folder are in outlier_terminal['terminal']
df = df[~df['sub_folder'].isin(outlier_terminal['terminal'])]


# In[9]:


#Data processing
df['sin_quarter'] = np.sin(2*np.pi*(df['QuarterOfTheYear']-0)/4)
df['cos_quarter'] = np.cos(2*np.pi*(df['QuarterOfTheYear']-0)/4)
df = df.drop(columns=['QuarterOfTheYear'])
target = list(df.columns.values)
target = [idx for idx in target if idx.lower().startswith('C_'.lower())]
#Parameters
end_training = '2019Q1'
begin_validation = '2018Q1'
end_validation = '2020Q1'
begin_testing = '2019Q1'
end_testing = '2020Q4'
n_past = 4 #Look back period
n_future = 4 #Number of period to forecast each time
batch_size = 1 #For creating dataset
n_epochs = 1000
df = df.set_index(['quarter'])
df = df.loc[df.index <= end_testing]
#Reorder columns
df = df[target + [col for col in df.columns if col not in target]]
n_unknown_features = len(target)
n_features = len([x for x in list(df.columns.values) if x not in ['quarter','sub_folder']])
n_deterministic_features = n_features-n_unknown_features
train_columns_list = [x for x in list(df.columns.values) if x not in ['quarter','sub_folder']]


# In[10]:


## Split and Scale
location_list = df[['sub_folder']].drop_duplicates()['sub_folder'].to_list()
train_df = []
val_df = []
test_df = []
for location in location_list:
    temp_df = df[df['sub_folder']==location].copy()
    temp_df = temp_df[train_columns_list]
    temp_train = temp_df.loc[:end_training].iloc[:-1]
    temp_val = temp_df.loc[begin_validation:end_validation].iloc[:-1]
    temp_test = temp_df.loc[begin_testing:]
    train_df.append(temp_train)
    val_df.append(temp_val)
    test_df.append(temp_test)
train_df = pd.concat(train_df)
test_df = pd.concat(test_df)
val_df = pd.concat(val_df)
true_labels = df[target+['sub_folder']][df[target+['sub_folder']].index.isin(test_df.index.unique()[n_past:].to_list())].copy()
true_lables_train = df[target+['sub_folder']][df[target+['sub_folder']].index.isin(train_df.index.unique()[n_past:].to_list())].copy()
true_labels_val = df[target+['sub_folder']][df[target+['sub_folder']].index.isin(val_df.index.unique()[n_past:].to_list())].copy()
# Scale the data
#Scale features
features = [x for x in list(train_df.columns.values) if x not in target]
#List of columns that start with stop or dwell
ais_features = [col for col in features if col.lower().startswith('stop'.lower()) or col.lower().startswith('dwell'.lower())]
feature_scaler = MinMaxScaler()
other_features = [x for x in features if x not in ais_features]
#Scale other features
train_df[other_features] = feature_scaler.fit_transform(train_df[other_features])
val_df[other_features] = feature_scaler.transform(val_df[other_features])
test_df[other_features] = feature_scaler.transform(test_df[other_features])
#Scale AIS features
ais_min = train_df[ais_features].min().min()
ais_max = train_df[ais_features].max().max()
train_df[ais_features] = (train_df[ais_features] - ais_min)/(ais_max - ais_min)
val_df[ais_features] = (val_df[ais_features] - ais_min)/(ais_max - ais_min)
test_df[ais_features] = (test_df[ais_features] - ais_min)/(ais_max - ais_min)
#Scale target
train_min = train_df[target].min().min()
train_max = train_df[target].max().max()
train_df[target] = (train_df[target] - train_min)/(train_max - train_min)
val_df[target] = (val_df[target] - train_min)/(train_max - train_min)
test_df[target] = (test_df[target] - train_min)/(train_max - train_min)
#Create dataset
val_windowed = create_dataset(val_df, n_deterministic_features, n_past, n_future,batch_size, target)
training_windowed = create_dataset(train_df, n_deterministic_features, n_past, n_future,batch_size,target)
test_windowed = create_dataset(test_df, n_deterministic_features, n_past, n_future,batch_size, target)


# In[11]:


# #For testing, Comment this out to run the model
# K.clear_session()
# #If model exists, del model
# if 'model' in locals():
#     del model
# if 'model' in locals():
#     print('Model exists')
# rn.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# tf.keras.utils.set_random_seed(SEED)
# tf.config.experimental.enable_op_determinism()
# 
# # Finds an embedding for the past
# past_inputs = Input(shape=(n_past, n_features), name='past_inputs')
# # Encoding the past
# n_layers = 512
# encoder = LSTM(n_layers, return_state=True)
# encoder_outputs, state_h, state_c = encoder(past_inputs)
# # Encoding the future
# future_inputs = Input(shape=(n_future, n_deterministic_features), name='future_inputs')
# # Combining future inputs with past output
# decoder_lstm = LSTM(n_layers, return_sequences=True)
# x = decoder_lstm(future_inputs, initial_state=[state_h, state_c])
# 
# #Hidden Layers:
# activation = 'tanh'
# n_hidden_layers = 1
# out_features = 32
# for i in range(n_hidden_layers):
#     x = Dense(out_features, activation=activation)(x)
#     out_features = int(out_features/2)
# 
# 
# output = Dense(len(target), activation='relu')(x)
# model = Model(inputs=[past_inputs, future_inputs], outputs=output)
# 
# optimizer_name = 'Adam'
# lr = 0.00001
# opt = getattr(optimizers, optimizer_name)(learning_rate=lr)
# 
# model.compile(loss=Huber(), optimizer=opt , metrics=['mae'])
# model.summary()
# 
# early_stopping = EarlyStopping(monitor='val_loss', patience=30, min_delta = 0, verbose=1,restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1, mode='min', min_delta = 0.0001)
# # checkpoint = ModelCheckpoint(filepath= par_dir / 'Outputs' / 'LSTM_Outputs' / (study_name + '_' + str(trial.number) + '.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch', options=None)
# model.fit(training_windowed,
#                         epochs=30,
#                         validation_data=val_windowed,
#                         shuffle=False,
#                         callbacks=[early_stopping, reduce_lr],
#                         verbose=2,
#                         batch_size=32)
# # Plot the loss and val_loss
# import matplotlib.pyplot as plt
# plt.plot(model.history.history['loss'], label='loss')
# plt.plot(model.history.history['val_loss'], label='val_loss')
# plt.legend() 
# plt.show() 
# print('Test_MAE:', model.evaluate(test_windowed))
# 
# actuals, forecasts = [], []
# for i, ((past, future), actual) in enumerate(test_windowed):
#     pred = model.predict((past, future), verbose=0, batch_size=1500)
#     actuals.append(actual.numpy().flatten())
#     forecasts.append(pred.flatten())
# print('Predict completed')
# #convert actuals to dataframe
# actuals_df = pd.DataFrame(actuals)
# #Convert to only one column starting from first row
# actuals_df = actuals_df.stack().reset_index(drop=True)
# actuals_df = pd.DataFrame(actuals_df, columns=['Actuals'])
# 
# # #Convert forecasts to dataframe
# forecasts_df = pd.DataFrame(forecasts)
# #Convert to only one column starting from first row
# forecasts_df = forecasts_df.stack().reset_index(drop=True)
# forecasts_df = pd.DataFrame(forecasts_df, columns=['Predictions'])
# 
# #Concatenate actuals and forecasts
# actuals_predictions_df = pd.concat([actuals_df, forecasts_df], axis=1)
# actuals_predictions_df
# 
# #Rescale actuals and predictions using train_min and train_max
# actuals_predictions_df['Actuals'] = (actuals_predictions_df['Actuals']*(train_max - train_min) + train_min)
# actuals_predictions_df['Predictions'] = (actuals_predictions_df['Predictions']*(train_max - train_min) + train_min)
# 
# #Set actuals to int
# actuals_predictions_df['Actuals'] = actuals_predictions_df['Actuals'].round(0).astype(int)
# #Create dataframe
# temp_label_df = []
# true_labels.reset_index(inplace=True)
# #Iterate through each row of true_labels
# for index, row in true_labels.iterrows():
#     quarter = row['quarter']
#     commodity = row[target]
#     sub_folder = row['sub_folder']
#     #Create dataframe using quarter, commodity, sub_folder
#     temp_df = pd.DataFrame({'quarter':quarter,'Commodity':target, 'Actuals':commodity,'sub_folder':sub_folder})
#     #Append to temp_label_df
#     temp_label_df.append(temp_df)
# #Concatenate temp_label_df
# temp_label_df = pd.concat(temp_label_df)
# #Set actuals to int
# temp_label_df['Actuals'] = temp_label_df['Actuals'].astype(int)
# temp_label_df = temp_label_df.reset_index().drop(columns=['index'])
# #Merge actuals_predictions_df with temp_label_df on index
# res = pd.merge(temp_label_df,actuals_predictions_df,how='left',left_index=True,right_index=True)
# 
# #Raise error if Actuals_x and Actuals_y are not equal
# if (res['Actuals_x'] != res['Actuals_y']).any():
#     raise ValueError('Actuals_x and Actuals_y are not equal')
# #Drop Actuals_y
# res = res.drop(columns=['Actuals_y'])
# #Rename Actuals_x to Actuals
# res = res.rename(columns={'Actuals_x':'Actuals'})


# In[ ]:


#Final Model
#Study name
study_name = data_model_dir[data_num].split('.')[0]
#Clear session and model
K.clear_session()
#If model exists, del model
if 'model' in locals():
    del model
if 'model' in locals():
    print('Model exists')
rn.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()
#LSTM Model
def define_model(trial):
    # Finds an embedding for the past
    past_inputs = Input(shape=(n_past, n_features), name='past_inputs')
    # Encoding the past
    n_layers = trial.suggest_categorical("n_layers", [32, 64, 128, 256, 512])
    encoder = LSTM(n_layers, return_state=True)
    encoder_outputs, state_h, state_c = encoder(past_inputs)
    # Encoding the future
    future_inputs = Input(shape=(n_future, n_deterministic_features), name='future_inputs')
    # Combining future inputs with past output
    decoder_lstm = LSTM(n_layers, return_sequences=True)
    x = decoder_lstm(future_inputs, initial_state=[state_h, state_c])

    #Hidden Layers:
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    n_hidden_layers = trial.suggest_categorical("n_hidden_layers", [1, 2])
    out_features = trial.suggest_categorical("n_units", [16, 32, 64])
    for i in range(n_hidden_layers):
        x = Dense(out_features, activation=activation)(x)
        out_features = int(out_features/2)

    output = Dense(len(target))(x)
    model = Model(inputs=[past_inputs, future_inputs], outputs=output)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    lr =0.00001
    opt = getattr(optimizers, optimizer_name)(learning_rate=lr)
    
    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            print('Duplicate parameters; Trial Pruned')
            raise optuna.exceptions.TrialPruned()

    model.compile(loss=Huber(), optimizer=opt , metrics=['mse'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, min_delta = 0, verbose=0,restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=200, min_lr=0.000001, verbose=0, mode='min', min_delta = 0.000001)
    checkpoint = ModelCheckpoint(filepath= par_dir / 'Outputs' / 'LSTM_Outputs' / (study_name + '_' + str(trial.number) + '.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch', options=None)

    csv_logger = CSVLogger(par_dir / 'Outputs' / 'LSTM_Outputs' / (study_name + '_' + str(trial.number) + '.csv'), separator=",", append=False)
    model.fit(training_windowed,
                        epochs=n_epochs,
                        validation_data=val_windowed,
                        shuffle=False,
                        callbacks=[early_stopping, checkpoint, csv_logger, reduce_lr],
                        verbose=0,
                        batch_size=32)

    #If file exists then load the model else return None
    if os.path.exists(par_dir / 'Outputs' / 'LSTM_Outputs' / (study_name + '_' + str(trial.number) + '.h5')):
        model = load_model(par_dir / 'Outputs' / 'LSTM_Outputs' / (study_name + '_' + str(trial.number) + '.h5'))
        score = model.evaluate(val_windowed, verbose=0)
        return score[1]
    else:
        return None

def pruning_callback(study, trial):
    # Check if the best objective value has improved in the last n_wait trials
    n_wait = 500
    if len(study.trials) < n_wait:
        return False

    # Get the best objective value found so far in the study
    best_value = study.best_value
    # Check if the best objective value has improved in the last n_look trials
    check_values = [
        t.value
        for t in study.trials[-n_wait:]
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if all(v > best_value for v in check_values):
        study.stop()
        print('Study Pruned')
    return False


# In[ ]:


#For Optuna
storage_path = "/home/USACE_Modeling/Outputs/LSTM_Outputs/LSTM_study.log" #For Slurm
# storage_path = par_dir / 'Outputs' / 'LSTM_Outputs' / 'LSTM_study.log' #For Local
storage_url = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(storage_path),)
search_space_lstm = {
    "n_layers": [32, 64, 128, 256, 512],
    "activation": ['relu', 'tanh'],
    "n_hidden_layers": [1, 2],
    "n_units": [16, 32, 64],
    "optimizer": ["Adam"],
}
#Get length of possible combinations
max_trial = 1
for key, value in search_space_lstm.items():
    max_trial = max_trial * len(value)
# max_trial = max_trial*2 #To account for repetitions
study = optuna.create_study(study_name = study_name, storage = storage_url, load_if_exists=True, direction='minimize', sampler=optuna.samplers.GridSampler(search_space = search_space_lstm ,seed = SEED))

if __name__ == "__main__":
    study = optuna.load_study(study_name = study_name, storage=storage_url)
    study.optimize(define_model,callbacks=[pruning_callback, MaxTrialsCallback(max_trial, states=(TrialState.COMPLETE,))], n_trials=1)

    #Keep the best model
    directory_path = par_dir / 'Outputs' / 'LSTM_Outputs' #For Local
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.number != study.best_trial.number:
            #Delete if file exists
            file_path = Path(directory_path) / (study_name + '_' + str(trial.number) + '.h5')
            if file_path.exists():
                file_path.unlink()
            csv_path = Path(directory_path) / (study_name + '_' + str(trial.number) + '.csv')
            if csv_path.exists():
                csv_path.unlink()


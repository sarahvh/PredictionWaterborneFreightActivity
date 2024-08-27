SEED = 1234
import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

# os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['PYTHONHASHSEED'] = str(SEED)
# os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from IPython.display import display
# random.seed(SEED)
import pandas as pd
import numpy as np

# np.random.seed(SEED)
import tensorflow as tf

# tf.random.set_seed(SEED)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#
# tf.get_logger().setLevel('ERROR')

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# if len(gpus) > 0:
#     tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)))

import matplotlib.pyplot as plt
from tqdm import tqdm


from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoTCN
from neuralforecast.models import TCN
from neuralforecast.losses.pytorch import HuberLoss

from ray import tune
# from PIL import ImageFont
import re
import datetime
import glob
import ast
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error

from functools import reduce
pd.set_option('display.float_format', lambda x: '%.3f' % x)

desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)
font = {'size': 14}


# def df_to_X_y(df, window_size, target):
#     df_as_np = df.to_numpy()
#     X = []
#     y = []
#     for i in range(len(df_as_np) - window_size):
#         row = [r for r in df_as_np[i:i + window_size]]
#         X.append(row)
#         label = []
#         for j in range(0, len(target)):
#             label.append(df_as_np[i + window_size][j])
#         y.append(label)
#     return np.array(X), np.array(y)


# def Xy_to_df(pred, act, m, index_label, target_scaler, target):
#     predictions = m.predict(pred)
#     pred_unscaled = predictions.copy()
#     predictions = target_scaler.inverse_transform(predictions)
#     actuals = target_scaler.inverse_transform(act)
#     target = [x[2:] for x in target]
#     app_df = []
#     for i in range(0, predictions.T.shape[0]):
#         df_t = pd.DataFrame(data={'Predictions': predictions.T[i], 'Actuals': actuals.T[i]})
#         df_t['Commodity'] = target[i]
#         app_df.append(df_t)
#     app_df = pd.concat(app_df)
#     index_cycle = itertools.cycle(index_label)
#     app_df['quarter'] = list(itertools.islice(index_cycle, len(app_df)))
#     return app_df, pred_unscaled

def create_dataset(df, n_deterministic_features, n_past, n_future, batch_size, target):
    total_size = n_past + n_future
    dfs = np.array_split(df, len(df) // (int(len(df)/int((len(df))/len(df.index.unique())))))
    append_data = []
    for i, df_split in enumerate(dfs):
        data = tf.data.Dataset.from_tensor_slices(df_split.values)
        data = data.window(total_size, shift=1, drop_remainder=True)
        data = data.flat_map(lambda k: k.batch(total_size))
        data = data.map(lambda k: ((k[:-n_future], k[-n_future:, -n_deterministic_features:]),
                                   k[-n_future:, 0:len(target)]))
        data = data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        append_data.append(data)
    # append_data = tf.data.Dataset.concatenate(*append_data)
    appended_data = reduce(lambda ds1, ds2: ds1.concatenate(ds2), append_data)
    return appended_data

def create_result_dataframe(forecasts, actuals, n_future, target_scaler, target):
    results_df = actuals.copy()
    for comm_var, j in zip(target, range(0,len(target))):
        arr = np.empty((0, len(forecasts) + n_future - 1), float)
        for idx, i in enumerate(forecasts):
            temp = np.empty(actuals.shape[0])
            temp[:] = np.nan
            temp[idx: n_future+idx] = i[j]
            arr = np.append(arr, temp.reshape(1, -1), axis=0)
        results_df[comm_var + '_pred'] = np.nanmean(arr.T, axis=1)
    pred_columns = results_df.columns.to_list()[-len(target):]
    results_df[pred_columns] = target_scaler.inverse_transform(results_df[pred_columns])
    return results_df, pred_columns

def commodity_threshold(df, threshold):
    comm_list = ['C_1','C_2','C_3','C_4','C_5','C_6','C_7','C_8','C_9']
    commodity_tonnage = df[comm_list+['sub_folder']].groupby('sub_folder').sum().reset_index().melt(id_vars=['sub_folder'],var_name='Commodity', value_vars=comm_list, value_name='Tonnage').sort_values('sub_folder')
    total_tonnage = commodity_tonnage.groupby(['sub_folder']).sum().reset_index().rename(columns={'Tonnage':'Total Tonnage'})
    commodity_tonnage = commodity_tonnage.merge(total_tonnage, on='sub_folder',how='left')
    commodity_tonnage['Percentage'] = commodity_tonnage['Tonnage']/commodity_tonnage['Total Tonnage']*100
    commodity_cols = commodity_tonnage[commodity_tonnage['Percentage']>=threshold][['sub_folder','Commodity']]
    return commodity_cols

def post_processing(df, cols_threshold, locations):
    melted_app = []
    for port_terminal in locations:
        cols = cols_threshold[cols_threshold['sub_folder'] == port_terminal]['Commodity'].to_list()
        cols.append('sub_folder')
        temp_df = df[df['sub_folder']==port_terminal].copy()
        temp_df = temp_df.filter(regex='^(' + '|'.join(cols) + ')')
        temp_df = temp_df.reset_index()

        # melt dataframe
        temp_df = temp_df.melt(id_vars=['quarter', 'sub_folder'], var_name='Metrics', value_name='Value')
        temp_df['Commodity'] = temp_df['Metrics'].str.extract(r'(C_\d+)')
        temp_df = temp_df[['quarter', 'Commodity', 'sub_folder', 'Metrics', 'Value']]

        # replace values in Metrics column
        temp_df['Commodity']=temp_df['Metrics']
        temp_df['Commodity'] = temp_df['Commodity'].str.replace('_pred', '').str.replace('_Error%', '').str.replace('_ErrorTon', '')
        temp_df.loc[temp_df['Commodity'] == temp_df['Metrics'], 'Metrics'] = 'Actual'
        temp_df.loc[temp_df['Metrics'].str.endswith('_pred'), 'Metrics'] = 'Predicted'
        temp_df['Metrics'] = temp_df['Metrics'].str.replace(r'^.*Error%', 'Error%').str.replace(r'^.*ErrorTon', 'ErrorTon')

        temp_df = temp_df.replace([np.inf, -np.inf], 'Infinity')

        temp_df = temp_df.pivot(index=['quarter','Commodity','sub_folder'], columns='Metrics',values='Value').reset_index()
        melted_app.append(temp_df)
    melted_app = pd.concat(melted_app)
    return melted_app

def total_commodities(df_total_commodities):
    # Creates dataframe without traffic type
    # Requires data with commodity split into domestic and foreign
    commodities = df_total_commodities.columns.to_list()
    commodities = [x for x in commodities if x.startswith('C_')]
    commodities = [x.replace('_domestic','') for x in commodities]
    commodities = [x.replace('_foreign','') for x in commodities]
    commodities = sorted(list(set(commodities)))

    temp_df = df_total_commodities.copy()
    for i in commodities:
        temp_df[i] = temp_df[i+'_domestic'] + temp_df[i+'_foreign']
        temp_df = temp_df.drop(columns=[i+'_domestic',i+'_foreign'])
    return temp_df

def data_df(df_temp, vessel_agg,ves_type, temporal_agg, features, traffic_type):
    temp_df = df_temp.copy()

    #For vessel Types
    if vessel_agg == 1:
        if ves_type == 'Self-Propelled, Dry Cargo':
            temp_df = temp_df[temp_df['dominant_vessel_type']==ves_type]
            # Keep columns that has word 'Cargo' or 'C_', or 'quarter', or 'QuarterOfTheYear' in temp_df
            pattern = r'\b\w*(?:'+'|'.join(['_Cargo_','C_','quarter','QuarterOfTheYear', 'sub_folder','folder'])+r')\w*\b'
            temp_df = temp_df[temp_df.columns[temp_df.columns.str.contains(pattern)]]

        elif ves_type == 'Self-Propelled, Tanker':
            temp_df = temp_df[temp_df['dominant_vessel_type']==ves_type]
            # Keep columns that has word 'Tanker' or 'C_', or 'quarter', or 'QuarterOfTheYear' in temp_df
            pattern = r'\b\w*(?:'+'|'.join(['_Tanker_','C_','quarter','QuarterOfTheYear', 'sub_folder','folder'])+r')\w*\b'
            temp_df = temp_df[temp_df.columns[temp_df.columns.str.contains(pattern)]]

        elif ves_type == 'Tug_Tow':
            temp_df = temp_df[temp_df['dominant_vessel_type']==ves_type]
            # Keep columns that has word 'Tug Tow' or 'C_', or 'quarter', or 'QuarterOfTheYear' in temp_df
            pattern = r'\b\w*(?:'+'|'.join(['_Tug Tow_','C_','quarter','QuarterOfTheYear', 'sub_folder','folder'])+r')\w*\b'
            temp_df = temp_df[temp_df.columns[temp_df.columns.str.contains(pattern)]]

        elif ves_type == 'Mixed':
            temp_df = temp_df[temp_df['dominant_vessel_type']==ves_type]
            pattern = r'\b\w*(?:'+'|'.join(['_Cargo_','_Tanker_','_Tow_Tug','C_','quarter','QuarterOfTheYear', 'sub_folder','folder'])+r')\w*\b'
            temp_df = temp_df[temp_df.columns[temp_df.columns.str.contains(pattern)]]


    # For temporal aggregation
    if temporal_agg == 'quarterly':
        #Remove columns that has '_M_' in temp_df
        pattern = r'\b\w*(?:'+'|'.join(['_M_'])+r')\w*\b'
        temp_df = temp_df[temp_df.columns[~temp_df.columns.str.contains(pattern)]]

    if temporal_agg == 'monthly':
        #Keep columns that has '_M' or 'quarter', or 'QuarterOfTheYear' or 'sub_folder' or 'folder' in temp_df
        pattern = r'\b\w*(?:'+'|'.join(['C_','_M_','quarter','QuarterOfTheYear','sub_folder','folder'])+r')\w*\b'
        temp_df = temp_df[temp_df.columns[temp_df.columns.str.contains(pattern)]]


    #For Filtering features
    # #Keep columns that has 'C_', or 'quarter', or 'QuarterOfTheYear' or elements in features in temp_df
    pattern = r'\b\w*(?:'+'|'.join(['C_','quarter','QuarterOfTheYear','sub_folder','folder']+features)+r')\w*\b'
    temp_df = temp_df[temp_df.columns[temp_df.columns.str.contains(pattern)]]

    if traffic_type == 0:
        temp_df = total_commodities(temp_df)

    return temp_df

import torch
import pandas as pd
import numpy as np

def time_features(dates, freq='h'):
    df = pd.DataFrame({'date': dates})
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['weekday'] = df.date.dt.weekday
    df['hour'] = df.date.dt.hour
    return df[['month', 'day', 'weekday', 'hour']].values

def prepare_data(df_masked):
    df_masked = df_masked.interpolate(limit_direction='both')
    values = df_masked.values.astype(np.float32)
    dates = pd.to_datetime(df_masked.index)
    time_feat = time_features(dates)

    x_enc = torch.tensor(values).unsqueeze(0)
    x_mark_enc = torch.tensor(time_feat).unsqueeze(0)
    x_dec = x_enc.clone()
    x_mark_dec = x_mark_enc.clone()

    return x_enc, x_mark_enc, x_dec, x_mark_dec, values


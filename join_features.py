import os
import pandas as pd
import pathlib
import gc
import numpy as np
import concurrent.futures
import time
from tqdm import tqdm
import numba as nb
import pandas as pd
import numpy as np
import sys
from fp_vecbacktest.vecbacktest.backtest import Backtest
from vecbacktest.operators import *
from vecbacktest.technical_utils import * 
from vecbacktest.statics.universes import FUTURES_UNIVERSE
from vecbacktest.base_alpha import BaseAlpha
import numba as nb
import argparse
import json
import pickle

def json_load(path):
    with open(path) as file:
        j = json.load(file)
    return j
      
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='/home/mp/quantdata/auxiliary/mp/qset/uniswap_liquidity_pools_resampler/configs/default_config.json',
                        help='configuration file')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='number of parallel jobs')
    args = parser.parse_args()
    args.config = json_load(args.config)
    return args

def merge_and_reindex_features(feature_name, instruments_list, start="2020-05-15 01:00:00", end="2024-09-01 00:00:00", freq="30T"):
    args = parse_arguments()
    feature_dir_template = args.config['feature_path']
    feature_dir = os.path.join(feature_dir_template, "{feature_name}")
    feature_dir = feature_dir.format(feature_name=feature_name)

    files = [f for f in os.listdir(feature_dir) if f.endswith(".parquet")]
    merged_filename = f"merged_{feature_name}.parquet"

    time_index = pd.date_range(start=start, end=end, freq=freq)
    merged_df = pd.DataFrame(index=time_index)

    for file in files:
        if file == merged_filename:
            continue

        instrument_name = file.replace(".parquet", "")
        file_path = os.path.join(feature_dir, file)

        df = pd.read_parquet(file_path)
        df = df.groupby('half_hour').max()
        df = df.rename(columns={"value": instrument_name[:-4] + '-' + instrument_name[-4:]})
        df = df.reindex(time_index)

        merged_df = merged_df.join(df, how="left")
        merged_df = merged_df[sorted(merged_df.columns)]

    merged_df = merged_df.reindex(columns = instruments_list)
    save_path = os.path.join(args.config['output'], f"merged_{feature_name}.parquet")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged_df.to_parquet(save_path)
    print(f"Объединённые данные сохранены в {save_path}")

    return merged_df

if __name__ == '__main__':
    config = {
        'data_folderpath': r'/data/ds/sandbox_insample_data',
        'universe': FUTURES_UNIVERSE,
        # 'liquid_universe': SPOT_UNIVERSE,
        'exec_prices': 'close',  # name of close fields for execution
        'feed_type': 'multi',  # multifolder backtesting, 'single' at default
        'execution_market': 'BINANCE-F',  # which folder contains execution prices
        'fundings': 'fundings',  # fundings field name, None at default
        'executor_config': {
            'fee': -0.00002,
            'type': 'constant',
            'params': {'const_spread': 10e-4}
            },
        'alpha_config': {
            'max_w': 0.05
        },
        'lazyload': True, # If False, data will be fully loaded to backtest object
    }
    
    bt = Backtest(config)
    features_to_load = [
    'close',
    'buy_volume', 
    'sell_volume'
    ]
    for feature_name in features_to_load:
        merge_and_reindex_features(feature_name, bt.data['STRATEX-BINANCE-F']['buy_trade_count'].columns)

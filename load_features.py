import pathlib
import pandas as pd
import os
import gc
import numpy as np
import concurrent.futures
import time
from tqdm import tqdm
import numba as nb
import json
from joblib import Parallel, delayed
import argparse

params = {}

params_quantile_1_ask = {
    'quantile': 1,
    'trade_side': 1,
    'trades_weights_feature_name': None
}
params_quantile_1_bid = {
    'quantile': 1,
    'trade_side': -1,
    'trades_weights_feature_name': None
}
params_ask = {
    'trade_side': 1,
}
params_bid = {
    'trade_side': -1,
}
params_ask_25_percentage = {
    'trade_side': 1,
    'percentage': 0.25,
}
params_bid_25_percentage = {
    'trade_side': -1,
    'percentage': 0.25,
}
params_ask_50_percentage = {
    'trade_side': 1,
    'percentage': 0.5,
}
params_bid_50_percentage = {
    'trade_side': -1,
    'percentage': 0.5,
}
params_ask_75_percentage = {
    'trade_side': 1,
    'percentage': 0.75,
}
params_bid_75_percentage = {
    'trade_side': -1,
    'percentage': 0.75,
}
params_ask_40_percentage = {
    'trade_side': 1,
    'percentage': 0.4,
}
params_bid_40_percentage = {
    'trade_side': -1,
    'percentage': 0.4,
}
params_ask_60_percentage = {
    'trade_side': 1,
    'percentage': 0.6,
}
params_bid_60_percentage = {
    'trade_side': -1,
    'percentage': 0.6,
}
params_ask_100_dollar_volume = {
    'trade_side': 1,
    'dollar_volume': 100,
}
params_bid_100_dollar_volume = {
    'trade_side': -1,
    'dollar_volume': 100,
}
params_ask_positive = {
    'trade_side': 1,
    'positive_checker': 1,
}
params_bid_positive = {
    'trade_side': -1,
    'positive_checker': 1,
}
params_ask_negative = {
    'trade_side': 1,
    'positive_checker': -1,
}
params_bid_negative = {
    'trade_side': -1,
    'positive_checker': -1,
}

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

def compute_big_volumes_volume_profile(link, params_dict):
    quantile = params_dict['quantile']
    trade_side = params_dict['trade_side']
    trades_weights_feature_name = params_dict['trades_weights_feature_name']

    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')

    df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)

    if trades_weights_feature_name is not None:
        df['qty'] *= df[trades_weights_feature_name]

    tmp = df.groupby(['half_hour', 'price']).agg({'qty': 'sum'}).reset_index()
    trades_data_volume_quantile = tmp.groupby('half_hour')['qty'].quantile(quantile).reset_index().rename(columns={'qty': 'qty_quantile'})
    tmp_new = pd.merge(tmp, trades_data_volume_quantile, on='half_hour')
    tmp_filter = tmp_new[tmp_new['qty'] >= tmp_new['qty_quantile']]
    count = tmp_filter.groupby(['half_hour']).agg({'qty': 'sum'}).fillna(0)

    result_series = pd.Series(count['qty'].values, index=pd.to_datetime(count.index, unit="ns"))

    del df, tmp, tmp_new, tmp_filter, count
    gc.collect()

    return result_series

@nb.njit()
def custom_ffill(data):
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            data[i] = data[i - 1] + 1
    return data

def compute_trades_in_row(link, params_dict):
        trade_side = params_dict['trade_side']
        
        df = pd.read_parquet(link)
        df = df.set_index('exch_ts').sort_index(kind='stable')
    
        df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)
    
        df = df.reset_index()
        df['trades_in_row'] = np.nan
        df.loc[df['side'] != df['side'].shift(1), 'trades_in_row'] = 1
        custom_ffill(df['trades_in_row'].values)
        
        df = df[df['side'] == trade_side]
        
        df = df.set_index('exch_ts')
        trades_data_group = df.groupby('half_hour').agg({'trades_in_row': 'max'})
        result_series = pd.Series(trades_data_group['trades_in_row'].values, index=pd.to_datetime(trades_data_group.index, unit="ns"))
        
        del df, trades_data_group
        gc.collect()
        
        return result_series

@nb.njit()
def custom_ffill_1(data, weights):
    for i in range(1, len(data)):
        if np.isnan(data[i]):
            data[i] = data[i - 1] + weights[i]
    return data

def compute_trades_time_in_row(link, params_dict):
    trade_side = params_dict['trade_side']
    
    df = pd.read_parquet(link)
    df = df.set_index('exch_ts').sort_index(kind='stable')

    df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)

    df = df.reset_index()

    time_diff = df['exch_ts'].diff()

    mask = df['side'] != df['side'].shift(1)

    df['trades_in_row'] = np.nan
    df.loc[mask, 'trades_in_row'] = time_diff[mask].values

    df = df[df['side'] == trade_side]

    df = df.set_index('exch_ts')
    trades_data_group = df.groupby('half_hour').agg({'trades_in_row': 'max'})

    result_series = pd.Series(trades_data_group['trades_in_row'].values, index=pd.to_datetime(trades_data_group.index, unit="ns"))

    del df, trades_data_group
    gc.collect()

    return result_series


@nb.njit()
def count_max_sum_volumes(volume_array, percentage):
    sorted_volumes = np.sort(volume_array)[::-1]
    sum_volumes = sorted_volumes.sum() * percentage
    sum_volume = 0
    i = 0
    while sum_volume < sum_volumes:
        sum_volume += sorted_volumes[i]
        i += 1
    return i

def compute_trades_in_volume_percentage(link, params_dict):
    percentage = params_dict['percentage']
    trade_side = params_dict['trade_side']
    trades_weights_feature_name = params_dict.get('trades_weights_feature_name', None)

    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')

    df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)

    if trades_weights_feature_name is not None:
        df['qty'] *= df[trades_weights_feature_name]

    grouped_volumes = df.groupby('half_hour')['qty'].apply(list)

    results = grouped_volumes.apply(lambda x: count_max_sum_volumes(np.array(x), percentage=percentage))

    result_series = pd.Series(results.values, index=pd.to_datetime(results.index, unit="ns"))

    del df, grouped_volumes, results
    gc.collect()

    return result_series

def compute_trades_with_big_dollar_volume_grouped(link, params_dict):
    dollar_volume = params_dict['dollar_volume']
    trade_side = params_dict['trade_side']
    trades_weights_feature_name = params_dict.get('trades_weights_feature_name', None)

    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')
    df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)
    df['dollar_volume'] = df['qty'] * df['price']
    tmp = df.groupby(['half_hour', 'exch_ts']).agg({'dollar_volume': 'sum'}).reset_index()
    tmp_filter = tmp[tmp['dollar_volume'] > dollar_volume]
    if trades_weights_feature_name is not None:
        count = tmp_filter.groupby('half_hour').agg({trades_weights_feature_name: 'sum'}).fillna(0)
        result_series = pd.Series(count[trades_weights_feature_name].values, index=pd.to_datetime(count.index, unit="ns"))
    else:
        count = tmp_filter['half_hour'].value_counts().sort_index()
        result_series = pd.Series(count.values, index=pd.to_datetime(count.index, unit="ns"))
    del df, tmp, tmp_filter, count
    gc.collect()
    return result_series

def compute_change_price_all_stack(link, params_dict):
    positive_checker = params_dict['positive_checker']
    trade_side = params_dict['trade_side']
    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')
    df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)
    df['diff'] = df['price'].diff()
    df_filtered = df[df['diff'] * positive_checker > 1e-8]
    count = df_filtered['half_hour'].value_counts().sort_index(kind='stable').fillna(0)
    result_series = pd.Series(count.values, index=pd.to_datetime(count.index, unit="ns"))
    del df, df_filtered, count
    gc.collect()
    return result_series

def compute_change_price_all_stack_grouped(link, params_dict):
    positive_checker = params_dict['positive_checker']
    trade_side = params_dict['trade_side']
    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')
    df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)
    df = df.reset_index()
    tmp = df.groupby(['half_hour', 'exch_ts']).agg({'price': 'mean'}).reset_index()
    tmp['diff'] = tmp['price'].diff()
    df_filtered = tmp[tmp['diff'] * positive_checker > 1e-8]
    count = df_filtered['half_hour'].value_counts().sort_index(kind='stable').fillna(0)
    result_series = pd.Series(count.values, index=pd.to_datetime(count.index, unit="ns"))
    del df, df_filtered, count
    gc.collect()
    return result_series

def compute_change_price_all_stack_grouped_1m(link, params_dict):
    positive_checker = params_dict['positive_checker']
    trade_side = params_dict['trade_side']
    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')
    df['half_hour'] = np.ceil(df.index.values / (60 * 1_000_000_000)) * (60 * 1_000_000_000)
    df = df.reset_index()
    tmp = df.groupby(['half_hour', 'exch_ts']).agg({'price': 'mean'}).reset_index()
    tmp['diff'] = tmp['price'].diff()
    df_filtered = tmp[tmp['diff'] * positive_checker > 1e-8]
    count = df_filtered['half_hour'].value_counts().sort_index(kind='stable').fillna(0)
    result_series = pd.Series(count.values, index=pd.to_datetime(count.index, unit="ns"))
    del df, df_filtered, count
    gc.collect()
    return result_series

def compute_trades_cnt_grouped(link, params_dict):
    trade_side = params_dict['trade_side']
    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')
    df['half_hour'] = np.ceil(df.index.values / (30 * 60 * 1_000_000_000)) * (30 * 60 * 1_000_000_000)
    df = df.reset_index()
    tmp = df.groupby(['half_hour', 'exch_ts']).agg({'price': 'mean'}).reset_index()
    count = tmp['half_hour'].value_counts().sort_index(kind='stable').fillna(0)
    result_series = pd.Series(count.values, index=pd.to_datetime(count.index, unit="ns"))
    del df, count
    gc.collect()
    return result_series



def compute_trades_cnt_grouped_1m(link, params_dict):
    trade_side = params_dict['trade_side']
    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')
    df['half_hour'] = np.ceil(df.index.values / (60 * 1_000_000_000)) * (60 * 1_000_000_000)
    df = df.reset_index()
    tmp = df.groupby(['half_hour', 'exch_ts']).agg({'price': 'mean'}).reset_index()
    count = tmp['half_hour'].value_counts().sort_index(kind='stable').fillna(0)
    result_series = pd.Series(count.values, index=pd.to_datetime(count.index, unit="ns"))
    del df, count
    gc.collect()
    return result_series


def compute_last_trade_price_half_hour(link, params_dict):
    trade_side = params_dict['trade_side']
    df = pd.read_parquet(link)
    df = df.set_index('exch_ts').sort_index(kind='stable')
    half_hour_ns = 30 * 60 * 1_000_000_000
    df['half_hour'] = np.ceil(df.index.values / half_hour_ns) * half_hour_ns
    df = df.reset_index()
    tmp = df.sort_values(['half_hour', 'exch_ts'], kind='stable').groupby('half_hour').agg({'price': 'last'})
    result_series = pd.Series(tmp['price'].values, index=pd.to_datetime(tmp.index, unit="ns"))
    del df, tmp
    gc.collect()
    return result_series


def compute_sum_volume_half_hour(link, params_dict):
    trade_side = params_dict['trade_side']
    df = pd.read_parquet(link)
    df = df[df['side'] == trade_side]
    df = df.set_index('exch_ts').sort_index(kind='stable')
    half_hour_ns = 30 * 60 * 1_000_000_000
    df['half_hour'] = np.ceil(df.index.values / half_hour_ns) * half_hour_ns
    df = df.reset_index()
    tmp = df.groupby('half_hour').agg({'size': 'sum'})
    result_series = pd.Series(tmp['size'].values, index=pd.to_datetime(tmp.index, unit="ns"))
    del df, tmp
    gc.collect()
    return result_series



def load_checkpoint(feature_name, CHECKPOINT_FILE):
    checkpoint_path = CHECKPOINT_FILE.format(feature_name=feature_name)

    if not os.path.exists(checkpoint_path):
        return set()

    try:
        with open(checkpoint_path, "r") as f:
            data = json.load(f)
            return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, ValueError):
        return set()

def save_checkpoint(instrument, feature_name, CHECKPOINT_FILE):
    checkpoint_path = CHECKPOINT_FILE.format(feature_name=feature_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    completed_instruments = load_checkpoint(feature_name, CHECKPOINT_FILE)
    completed_instruments.add(instrument)

    with open(checkpoint_path, "w") as f:
        json.dump(list(completed_instruments), f)

def process_instrument(instrument, x, process_file, params_dict, feature_name):
    feature_dir = os.path.join(BASE_FEATURES_DIR, feature_name)
    save_path = os.path.join(feature_dir, f"{instrument}.parquet")

    completed_instruments = load_checkpoint(feature_name, CHECKPOINT_FILE)
    if instrument in completed_instruments:
        print(f"{instrument} уже обработан, пропускаем...")
        return instrument, None

    data = pd.Series(dtype="float64")
    instr_links = sorted([link for link in x if instrument in str(link)])

    all_data = Parallel(n_jobs=-1)(
        delayed(process_file)(link, params_dict) for link in tqdm(instr_links, desc=f"Processing {instrument}")
    )
    all_data = [d for d in all_data if d is not None]

    if all_data:
        data = pd.concat(all_data, ignore_index=False)
        del all_data
        gc.collect()

        os.makedirs(feature_dir, exist_ok=True)
        data.to_frame(name="value").to_parquet(save_path, index=True)
        print(f"Данные для {instrument} сохранены в {save_path}")

        save_checkpoint(instrument, feature_name, CHECKPOINT_FILE)

    return instrument, data

def process_all_instruments(instruments_list, x, process_file, params_dict, feature_name, CHECKPOINT_FILE, max_workers=4):
    start_time = time.time()
    completed_instruments = load_checkpoint(feature_name, CHECKPOINT_FILE)
    remaining_instruments = [inst for inst in instruments_list if inst not in completed_instruments]

    results = Parallel(n_jobs=max_workers)(
        delayed(process_instrument)(instrument, x, process_file, params_dict, feature_name)
        for instrument in tqdm(remaining_instruments, desc="Processing Instruments")
    )

    gc.collect()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nВремя выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")



if __name__ == '__main__':
    args = parse_arguments()
    BASE_FEATURES_DIR = args.config['feature_path']
    CHECKPOINT_FILE = os.path.join(BASE_FEATURES_DIR, "{feature_name}/checkpoint.json")
    print(CHECKPOINT_FILE)
    data_path = pathlib.Path(args.config['raw_data_path'])
    x = list(data_path.glob('*.parquet'))
    inst_list = []
    for link in x:
        file_name = os.path.basename(link)
        instrument = file_name.split("_")[0]
        inst_list.append(instrument)
    instruments_list_dirty = sorted(list(set(inst_list)))
    instruments_list = []
    for instrument in instruments_list_dirty:
        if instrument not in ['BTCUSDT-240329', 'BTCUSDT-240628', 'BTCUSDT-240927', 'BTCUSDT-241227', 
                              'ETHUSDT-240329', 'ETHUSDT-240628', 'ETHUSDT-240927', 'ETHUSDT-241227']:
            instruments_list.append(instrument)      
    instruments_list.sort()
    x.sort()
    process_all_instruments(instruments_list, x, compute_last_trade_price_half_hour, params, 'close', CHECKPOINT_FILE, max_workers = 10)
    process_all_instruments(instruments_list, x, compute_sum_volume_half_hour, params_bid, 'buy_volume', CHECKPOINT_FILE, max_workers = 10)
    process_all_instruments(instruments_list, x, compute_sum_volume_half_hour, params_ask, 'sell_volume', CHECKPOINT_FILE, max_workers = 10)

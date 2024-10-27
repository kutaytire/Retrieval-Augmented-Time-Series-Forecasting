from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import datasets
from gluonts.dataset.split import split
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import batcher
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Optional, Union
import os
import matplotlib.pyplot as plt

import numpy as np
from gluonts.dataset.arrow import ArrowWriter

def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    train_size = backtest_config["train_size"]
    valid_size = backtest_config["valid_size"]
    test_size = backtest_config["test_size"]

    trust_remote_code = True 

    ds = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )
    ds.set_format("numpy")
    df = ds.to_pandas()

    train_df, remaining_df = train_test_split(df, train_size=train_size, random_state=42)
    valid_df, test_df = train_test_split(remaining_df, test_size=test_size / (valid_size + test_size), random_state=42)

    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(valid_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, valid_df, test_df

def normalize_data(df, mean, std):
    df.loc[:, 'target'] = (df['target'] - mean) / std
    return df

def denormalize_data(df, mean, std):
    df.loc[:, 'target'] = df['target'] * std + mean
    return df

def ensure_numeric(df, column):
    df[column].fillna(0, inplace=True)  # Fill missing values with 0
    return df

def convert_to_arrow(
    path: Union[str, Path],
    time_series: Union[List[np.ndarray], np.ndarray],
    start_times: Optional[Union[List[np.datetime64], np.ndarray]] = None,
    compression: str = "lz4",
):
    if start_times is None:
        # Set an arbitrary start time
        start_times = [np.datetime64("2000-01-01 00:00", "s")] * len(time_series)

    assert len(time_series) == len(start_times)

    dataset = [
        {"start": start, "target": ts} for ts, start in zip(time_series, start_times)
    ]
    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=path,
    )

def normalize_segment(segment):
    mean = segment.mean()
    std = segment.std() + 1e-5
    normalized_segment = (segment - mean) / std
    return normalized_segment

def min_max_scale(series, min_val, max_val):
    series_min = series.min()
    series_max = series.max()
    return (series - series_min) / (series_max - series_min) * (max_val - min_val) + min_val

def plot_with_formatting(dataset_name, idx, past_values, true_values, baseline_forecast, retrieve_forecast, x_values, plot_save_dir):
    
    combined_past_and_true = np.concatenate([past_values, true_values])
    fig, axes = plt.subplots(1, 2, figsize=(16, 2))  
    
    # Plot 1: Baselime Forecast on the first subplot
    axes[0].plot(x_values[:len(combined_past_and_true)], combined_past_and_true, label='Ground Truth', color='#1f77b4', linewidth=2.0)
    forecast_time = x_values[len(past_values):len(past_values) + len(baseline_forecast)]
    axes[0].plot(forecast_time, baseline_forecast, label='Baseline', color='#e37777', linestyle='-', linewidth=3.0)
    axes[0].axvline(x=x_values[len(past_values)], color='gray', linestyle='--', linewidth=1)
    axes[0].set_title(f"{dataset_name} - Baseline", fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True)

    # Plot 2: Retrieved Forecast on the second subplot
    axes[1].plot(x_values[:len(combined_past_and_true)], combined_past_and_true, label='Ground Truth', color='#1f77b4', linewidth=2.0)
    axes[1].plot(forecast_time, retrieve_forecast, label='RAF', color='#FFA500', linestyle='-', linewidth=3.0)
    axes[1].axvline(x=x_values[len(past_values)], color='gray', linestyle='--', linewidth=1)
    axes[1].set_title(f"{dataset_name} - RAF", fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True)

    plt.tight_layout()

    plot_filename_combined = os.path.join(plot_save_dir, f"{dataset_name}_forecast_{idx + 1}_combined.png")
    plot_filename_combined_pdf = os.path.join(plot_save_dir, f"{dataset_name}_forecast_{idx + 1}_combined.pdf")
    plt.savefig(plot_filename_combined, format='png', bbox_inches='tight')
    plt.savefig(plot_filename_combined_pdf, format='pdf', bbox_inches='tight')
    plt.close()

    return plot_filename_combined



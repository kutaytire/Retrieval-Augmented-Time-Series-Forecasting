import logging
from pathlib import Path
from typing import Iterable, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.itertools import batcher
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from time_series_utils import augment_time_series, denormalize_predictions, normalize_context
import sys
import matplotlib.pyplot as plt
import os
from data_utils import plot_with_formatting

# Add the path to the folder containing the model to sys.path
from chronos_local import ChronosPipeline


app = typer.Typer(pretty_exceptions_enable=False)


# Taken from pandas._libs.tslibs.dtypes.OFFSET_TO_PERIOD_FREQSTR
offset_alias_to_period_alias = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q",
    "BQE-JAN": "Q",
    "BQE-FEB": "Q",
    "BQE-MAR": "Q",
    "BQE-APR": "Q",
    "BQE-MAY": "Q",
    "BQE-JUN": "Q",
    "BQE-JUL": "Q",
    "BQE-AUG": "Q",
    "BQE-SEP": "Q",
    "BQE-OCT": "Q",
    "BQE-NOV": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "Y": "Y",
    "BYE": "Y",
    "BYE-DEC": "Y",
    "BYE-JAN": "Y",
    "BYE-FEB": "Y",
    "BYE-MAR": "Y",
    "BYE-APR": "Y",
    "BYE-MAY": "Y",
    "BYE-JUN": "Y",
    "BYE-JUL": "Y",
    "BYE-AUG": "Y",
    "BYE-SEP": "Y",
    "BYE-OCT": "Y",
    "BYE-NOV": "Y",
    "YS": "Y",
    "BYS": "Y",
    "QS-JAN": "Q",
    "QS-FEB": "Q",
    "QS-MAR": "Q",
    "QS-APR": "Q",
    "QS-MAY": "Q",
    "QS-JUN": "Q",
    "QS-JUL": "Q",
    "QS-AUG": "Q",
    "QS-SEP": "Q",
    "QS-OCT": "Q",
    "QS-NOV": "Q",
    "QS-DEC": "Q",
    "BQS-JAN": "Q",
    "BQS-FEB": "Q",
    "BQS-MAR": "Q",
    "BQS-APR": "Q",
    "BQS-MAY": "Q",
    "BQS-JUN": "Q",
    "BQS-JUL": "Q",
    "BQS-AUG": "Q",
    "BQS-SEP": "Q",
    "BQS-OCT": "Q",
    "BQS-NOV": "Q",
    "BQS-DEC": "Q",
    "YS-JAN": "Y",
    "YS-FEB": "Y",
    "YS-MAR": "Y",
    "YS-APR": "Y",
    "YS-MAY": "Y",
    "YS-JUN": "Y",
    "YS-JUL": "Y",
    "YS-AUG": "Y",
    "YS-SEP": "Y",
    "YS-OCT": "Y",
    "YS-NOV": "Y",
    "YS-DEC": "Y",
    "BYS-JAN": "Y",
    "BYS-FEB": "Y",
    "BYS-MAR": "Y",
    "BYS-APR": "Y",
    "BYS-MAY": "Y",
    "BYS-JUN": "Y",
    "BYS-JUL": "Y",
    "BYS-AUG": "Y",
    "BYS-SEP": "Y",
    "BYS-OCT": "Y",
    "BYS-NOV": "Y",
    "BYS-DEC": "Y",
    "Y-JAN": "Y-JAN",
    "Y-FEB": "Y-FEB",
    "Y-MAR": "Y-MAR",
    "Y-APR": "Y-APR",
    "Y-MAY": "Y-MAY",
    "Y-JUN": "Y-JUN",
    "Y-JUL": "Y-JUL",
    "Y-AUG": "Y-AUG",
    "Y-SEP": "Y-SEP",
    "Y-OCT": "Y-OCT",
    "Y-NOV": "Y-NOV",
    "Y-DEC": "Y-DEC",
    "Q-JAN": "Q-JAN",
    "Q-FEB": "Q-FEB",
    "Q-MAR": "Q-MAR",
    "Q-APR": "Q-APR",
    "Q-MAY": "Q-MAY",
    "Q-JUN": "Q-JUN",
    "Q-JUL": "Q-JUL",
    "Q-AUG": "Q-AUG",
    "Q-SEP": "Q-SEP",
    "Q-OCT": "Q-OCT",
    "Q-NOV": "Q-NOV",
    "Q-DEC": "Q-DEC",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "W-SUN": "W-SUN",
}


def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_freq = pd.infer_freq(hf_dataset[0]["timestamp"])
    dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamp"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    assert len(gts_dataset) == dataset_length

    return gts_dataset

def get_test_sequences(test_input, test_len):
    test_instance_list = []
    for instance in test_input:
        test_instance = instance["target"][-test_len:]
        test_values = test_instance.astype(float)  
        test_values_tensor = torch.tensor(test_values)
        test_instance_list.append(test_values_tensor)

    return test_instance_list


def load_and_split_dataset(backtest_config: dict):
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]
    max_history = backtest_config["max_history"]
    distance = backtest_config["distance"]

    # This is needed because the datasets in autogluon/chronos_datasets_extra cannot
    # be distribued due to license restrictions and must be generated on the fly
    trust_remote_code = True if hf_repo == "autogluon/chronos_datasets_extra" else False

    ds = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )
    ds.set_format("numpy")

    gts_dataset = to_gluonts_univariate(ds)
    train_df, remaining_df = train_test_split(gts_dataset, train_size=0.7, random_state=42)
    _, test_df = train_test_split(remaining_df, test_size=0.2 / (0.1 + 0.2), random_state=42)

    # Split dataset for evaluation
    _, test_template = split(test_df, offset=offset)   
    test_data = test_template.generate_instances(prediction_length, windows=num_rolls, distance=distance, max_history=max_history)
    
    return test_data, train_df


def generate_sample_forecasts(
    train_df,
    augment: bool,
    top_n: int,
    test_data_input: Iterable,
    pipeline: ChronosPipeline,
    prediction_length: int,
    batch_size: int,
    num_samples: int,
    **predict_kwargs,
):
    # Generate forecast samples
    forecast_samples = []
    for batch in tqdm(batcher(test_data_input, batch_size=batch_size)):
        context = [torch.tensor(entry["target"]) for entry in batch]

        if augment:
            context,mean_std_values = augment_time_series(train_df, pipeline, context, prediction_length, top_n)
        else:
            context,mean_std_values = normalize_context(context)
        
        prediction = pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples,
            limit_prediction_length=False,
            **predict_kwargs,
        ).cpu()
        prediction = denormalize_predictions(prediction, mean_std_values)
        forecast_samples.append(prediction)
        
    # Convert forecast samples into gluonts SampleForecast objects
    sample_forecasts = []
    forecast_samples = np.concatenate(forecast_samples)
       
    for item, ts in zip(forecast_samples, test_data_input):
        forecast_start_date = ts["start"] + len(ts["target"])
        sample_forecasts.append(
            SampleForecast(samples=item, start_date=forecast_start_date)
        )

    return sample_forecasts


@app.command()
def main(
    config_path: Path,
    chronos_model_id: str = "amazon/chronos-t5-base",
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    batch_size: int = 75,
    num_samples: int = 20,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    augment: bool = False,
    top_n: int = 1,
):
    if isinstance(torch_dtype, str):
        torch_dtype = getattr(torch, torch_dtype)
    assert isinstance(torch_dtype, torch.dtype)

    # Load backtest configs
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    for config in backtest_configs:
        pipeline = ChronosPipeline.from_pretrained(
        chronos_model_id,
        device_map=device,
        torch_dtype=torch_dtype,
        )

        dataset_name = config["name"]
        prediction_length = config["prediction_length"]
        
        logger.info(f"Loading {dataset_name}")
        test_data, train_df = load_and_split_dataset(backtest_config=config)

        logger.info(
            f"Generating forecasts for {dataset_name} "
            f"({len(test_data.input)} time series)"
        )
        baseline_sample_forecasts = generate_sample_forecasts(
            train_df,
            augment, 
            top_n,
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        raf_sample_forecasts = generate_sample_forecasts(
            train_df,
            not augment, 
            top_n,
            test_data.input,
            pipeline=pipeline,
            prediction_length=prediction_length,
            batch_size=batch_size,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        true_values = []  
        past_values = []  
        past_time_values = []  
        true_time_values = []  

        for data_point in test_data.input:
            past_values.append(data_point['target'])  
            start_time = data_point['start']
            
            time_periods = pd.period_range(start=start_time, periods=len(data_point['target']), freq=start_time.freq)
            past_time_values.append(time_periods.to_timestamp())  

        
        for data_point in test_data.label:
            true_values.append(data_point['target'])  
            start_time = data_point['start']
            
            time_periods = pd.period_range(start=start_time, periods=len(data_point['target']), freq=start_time.freq)
            true_time_values.append(time_periods.to_timestamp())  # Convert to timestamps

        true_values = np.array(true_values)
        past_values = np.array(past_values)

        for idx, forecast in enumerate(baseline_sample_forecasts):

            baseline_forecast = np.mean(forecast.samples, axis=0)
            raf_forecast = np.mean(raf_sample_forecasts[idx].samples, axis=0)

            past_time = past_time_values[idx]
            true_time = true_time_values[idx]

            plot_save_dir = Path(f"paper_plots/{dataset_name}")
            plot_save_dir.mkdir(parents=True, exist_ok=True)
            
            _ = plot_with_formatting(
                dataset_name=dataset_name,         
                idx=idx,                           
                past_values=past_values[idx],    
                true_values=true_values[idx],      
                baseline_forecast=baseline_forecast,  
                retrieve_forecast=raf_forecast,     
                x_values=np.concatenate([past_time, true_time]),  
                plot_save_dir=str(plot_save_dir)      
            )

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("Chronos Plotting")
    logger.setLevel(logging.INFO)
    app()
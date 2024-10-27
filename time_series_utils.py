import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

def find_best_matches_full_series_batch(train_df, context_tensor_matrix, test_length, prediction_length, pipeline, top_n=1):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare context embeddings
    context_batch_tensor = torch.stack(context_tensor_matrix).to(device)
    target_embeddings, _ = pipeline.embed(context_batch_tensor)
    del context_batch_tensor
    torch.cuda.empty_cache()

    target_embeddings = target_embeddings.unsqueeze(1)  

    step_size = 1
    batch_size = 10

    all_errors = []
    all_indices = []
    idx = 0

    for start_idx in tqdm(range(0, len(train_df), batch_size), total=int(len(train_df)/batch_size)):
        end_idx = min(start_idx + batch_size, len(train_df))
        current_series_batch = train_df[start_idx:end_idx].copy()

        segment_list = []
        indices_list = []

        for series in current_series_batch:
            series_values = series["target"].copy()
            series_length = len(series_values)
            for i in range(0, series_length - test_length - prediction_length + 1, step_size):
                segment_values = series_values[i:i+test_length].copy()
                segment_values = segment_values.astype(float)
                segment_tensor = torch.tensor(segment_values).to(device)

                segment_list.append(segment_tensor)
                indices_list.append((idx, i))
            idx += 1

        if not segment_list:
            continue

        batch_tensor = torch.stack(segment_list).to(device)
        other_embeddings, _ = pipeline.embed(batch_tensor)
        del segment_list, batch_tensor
        torch.cuda.empty_cache()

        other_embeddings = other_embeddings.unsqueeze(0)    
        error_matrix = torch.norm(target_embeddings - other_embeddings, dim=3, p=2)
        errors = error_matrix.sum(dim=2)

        all_errors.append(errors.cpu())  
        all_indices.extend(indices_list)

        del other_embeddings, error_matrix, errors
        torch.cuda.empty_cache()

    if not all_errors:
        return []

    all_errors = torch.cat(all_errors, dim=1)  
    best_matches = []

    for target_idx, target_errors in enumerate(all_errors):
        top_n_errors, top_n_indices = torch.topk(target_errors, top_n, largest=False)
        for i in range(top_n):
            min_error_idx = top_n_indices[i].item()
            min_error = top_n_errors[i].item()
            series_idx, index = all_indices[min_error_idx]
            best_matches.append((series_idx, min_error, index))
    del target_embeddings

    future_and_last_segment = test_length + prediction_length
    best_match_segments = []
    for series_idx, min_distance, best_match_index in best_matches:
        matching_series = train_df[series_idx]["target"]
        best_match_segment = matching_series[best_match_index:(best_match_index + future_and_last_segment)]
        best_match_segments.append(best_match_segment)

    return best_match_segments

def augment_time_series(train_df, pipeline, context_tensor_matrix, prediction_length, top_n):
    test_length = len(context_tensor_matrix[0])
    best_matches = find_best_matches_full_series_batch(train_df, context_tensor_matrix, test_length, prediction_length, pipeline, top_n)

    cnt = 0
    augmented_matrix = []
    mean_std_values = []
    for context_tensor in context_tensor_matrix:
        context_tensor = torch.tensor(context_tensor, dtype=torch.float32)
        elements = best_matches[cnt:cnt+top_n]
        avg_best_segment = np.mean(elements, axis=0)
        avg_segment_tensor = torch.tensor(avg_best_segment)
        
        mask = ~torch.isnan(avg_segment_tensor)
        avg_mean = avg_segment_tensor[mask].mean()
        avg_std = torch.sqrt(((avg_segment_tensor[mask] - avg_mean) ** 2).mean()) + 1e-7
        avg_segment_tensor = normalize(avg_segment_tensor, avg_mean, avg_std)

        mask = ~torch.isnan(context_tensor)
        context_mean = context_tensor[mask].mean()
        context_std = torch.sqrt(((context_tensor[mask] - context_mean) ** 2).mean()) + 1e-7
        context_tensor = normalize(context_tensor, context_mean, context_std)        

        if np.isnan(context_tensor[0].numpy()):
            for elem in context_tensor[1:]:
                if not np.isnan(elem):
                    context_start = elem
                    break
        else:
            context_start = context_tensor[0]
        if torch.isnan(context_tensor).all():
            context_start = 0  
        best_segment_start = avg_segment_tensor[-1].numpy()
       
        difference = context_start - best_segment_start
        avg_segment_tensor += difference
        
        augmented_tensor = torch.cat((avg_segment_tensor, context_tensor))
        augmented_matrix.append(augmented_tensor)
        mean_std_values.append((context_mean, context_std))

        cnt += top_n
        
    return augmented_matrix, mean_std_values

def augment_time_series_fine_tune(train_df, pipeline, context_tensor_matrix, prediction_length, top_n):
    test_length = len(context_tensor_matrix[0])
    best_matches = find_best_matches_full_series_batch(train_df, context_tensor_matrix, test_length, prediction_length, pipeline, top_n)

    cnt = 0
    augmented_matrix = []
    mean_std_values = []
    for context_tensor in context_tensor_matrix:
        context_tensor = torch.tensor(context_tensor, dtype=torch.float32)
        elements = best_matches[cnt:cnt+top_n]
        avg_best_segment = np.mean(elements, axis=0)
        avg_segment_tensor = torch.tensor(avg_best_segment)
        
        mask = ~torch.isnan(avg_segment_tensor)
        avg_mean = avg_segment_tensor[mask].mean()
        avg_std = torch.sqrt(((avg_segment_tensor[mask] - avg_mean) ** 2).mean()) + 1e-7
        avg_segment_tensor = normalize(avg_segment_tensor, avg_mean, avg_std)

        mask = ~torch.isnan(context_tensor)
        context_mean = context_tensor[mask].mean()
        context_std = torch.sqrt(((context_tensor[mask] - context_mean) ** 2).mean()) + 1e-7
        context_tensor = normalize(context_tensor, context_mean, context_std)        

        if np.isnan(context_tensor[0].numpy()):
            for elem in context_tensor[1:]:
                if not np.isnan(elem):
                    context_start = elem
                    break
        else:
            context_start = context_tensor[0]

        if torch.isnan(context_tensor).all():
            context_start = 0  
        best_segment_start = avg_segment_tensor[-1].numpy()
    
        difference = context_start - best_segment_start
        avg_segment_tensor += difference
        
        augmented_tensor = torch.cat((avg_segment_tensor ,context_tensor))
        augmented_matrix.append(augmented_tensor)
        mean_std_values.append((context_mean, context_std))

        cnt += top_n
        
    return augmented_matrix, mean_std_values

def min_max_scale(tensor, min_val, max_val):
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    scaled_tensor = (tensor - tensor_min) / (tensor_max - tensor_min) * (max_val - min_val) + min_val
    return scaled_tensor

def normalize(tensor, mean, std):
    return (tensor - mean) / std

def denormalize_predictions(predictions, mean_std_values):
    denormalized_predictions = []
    for idx, prediction in enumerate(predictions):
        mean, std = mean_std_values[idx]
        prediction = prediction * std + mean
        prediction = torch.nan_to_num(prediction, nan=0.0)
        denormalized_predictions.append(prediction.cpu().numpy())

    return denormalized_predictions

def normalize_context(context_tensor_matrix):
    mean_std_values = []
    normalized_context = []
    for idx, context_tensor in enumerate(context_tensor_matrix):
        context_tensor = torch.tensor(context_tensor, dtype=torch.float32)
      
        mask = ~torch.isnan(context_tensor)
        context_mean = context_tensor[mask].mean()
        context_std = torch.sqrt(((context_tensor[mask] - context_mean) ** 2).mean()) + 1e-7
        context_tensor = normalize(context_tensor, context_mean, context_std)    
        
        normalized_context.append(context_tensor)
        mean_std_values.append((context_mean, context_std))

    return normalized_context, mean_std_values



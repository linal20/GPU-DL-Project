from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
from sklearn.metrics import (make_scorer, mean_squared_error, accuracy_score, 
                            f1_score, recall_score, precision_score, 
                            balanced_accuracy_score, matthews_corrcoef,
                            confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

min_penalty_val = 1.2
max_penalty_val = 3

def remove_cens(X_train,y_train,y2_train,t0_train,predicted_indices,d,d1,d2):
    X_train_D=X_train[predicted_indices!=0,:]
    y_train_D=y_train[predicted_indices!=0,]
    y2_train_D=y2_train[predicted_indices!=0,]
    t0_train_D=t0_train[predicted_indices!=0,]
    d_D=d[predicted_indices!=0,]
    d1_D=d1[predicted_indices!=0,]
    d2_D=d2[predicted_indices!=0,]
    # predicted_indices_D=predicted_indices[predicted_indices!=0]
    return([X_train_D,y_train_D,y2_train_D,t0_train_D,d_D,d1_D,d2_D])

import imblearn
from imblearn.over_sampling import SMOTE
# def resample(X_train,y_train,y2_train, d, include = 'd'):
#     y_train = y_train[:, np.newaxis]  # Or y_train.reshape(-1, 1)
#     y2_train = y2_train[:, np.newaxis]
#     d = d[:, np.newaxis]
#     smote = SMOTE()
    
#     if include == 'd':
#          # Stack X_train and X2 for resampling
#         X_combined = np.hstack((X_train, y_train, y2_train, d))
    
#         smote = SMOTE(random_state=42)
#         X_resampled, d_resampled= smote.fit_resample(X_combined, d)
     
#     else:
#         # Stack X_train and X2 for resampling
#         X_combined = np.hstack((X_train, y_train, y2_train))
        
#         smote = SMOTE(random_state=42)
#         X_resampled, y_train_resampled= smote.fit_resample(X_combined, y_train)

#         d_resampled = np.empty((0,))

#     # Split X_resampled back into original components
#     X_train_resampled = X_resampled[:, :X_train.shape[1]]
#     y_train_resampled = X_resampled[:, X_train.shape[1]:(X_train.shape[1]+ y_train.shape[1])].squeeze(1)
#     y2_train_resampled = X_resampled[:, (X_train.shape[1]+ y_train.shape[1]):(X_train.shape[1]+ y_train.shape[1] + y2_train.shape[1])].squeeze(1)
    
#     return([X_train_resampled,y_train_resampled,y2_train_resampled, d_resampled])



def mod_training(mod, X_train_fit, y_train_fit, X_train, X_test):
  
    mod.fit(X_train_fit, y_train_fit)
    
    # y_pred2 is classification prediction
    y_pred = mod.predict(X_train)
    y_pred_test = mod.predict(X_test)
    return ([y_pred, y_pred_test])


def mod_training(mod, X_train_fit, y_train_fit, X_train, X_test):
  
    mod.fit(X_train_fit, y_train_fit)
    
    # y_pred2 is classification prediction
    y_pred = mod.predict(X_train)
    y_pred_test = mod.predict(X_test)
    return ([y_pred, y_pred_test])


def convert_arr(arr):
    if np.array_equal(np.unique(arr), [0., 1.]):
        arr[arr == 0.] = 2.
    return(arr)

    
def accuracy_analysis(y_pred2, y_pred2_test, y_train, y_test, step=1):
    # Apply the conversion only if step == 2
    if step == 2:
        y_pred2 = convert_arr(y_pred2)
        y_pred2_test = convert_arr(y_pred2_test)

    print(f"step: {step}")
    # Print predictions for debugging
    print(y_pred2)
    
    # Calculate and print training accuracy
    acc = (y_pred2 == y_train.astype(np.int64)).sum().item() / len(y_train)
    acc = float(format(acc, '.3f'))
    print("accuracy train: ", acc)
    
    # Print confusion matrix for training data
    eval_matrix = pd.crosstab(y_train, y_pred2)
    print(eval_matrix)
    
    # Calculate and print testing accuracy
    acc_test = (y_pred2_test == y_test.astype(np.int64)).sum().item() / len(y_test)
    acc_test = float(format(acc_test, '.3f'))
    print("accuracy test: ", acc_test)

    # Print confusion matrix for training data
    eval_matrix = pd.crosstab(y_test, y_pred2_test)
    print(eval_matrix)

    # Calculate and print balanced accuracy
    bacc = balanced_accuracy_score(y_train, y_pred2)
    bacc_test = balanced_accuracy_score(y_test, y_pred2_test)
    print("Balanced Accuracy for Training:", bacc)
    print("Balanced Accuracy for Testing set:", bacc_test)

    # Calculate and print Matthews correlation coefficient (MCC)
    mcc = matthews_corrcoef(y_pred2, y_train)
    mcc_test = matthews_corrcoef(y_pred2_test, y_test)
    print("mcc for Training:", mcc)
    print("mcc for Testing set:", mcc_test)
    
    # Print confusion matrix for testing data
    eval_matrix = pd.crosstab(y_test, y_pred2_test)
    print(eval_matrix)

    # Return all metrics as a list
    return [acc, acc_test, bacc, bacc_test, mcc, mcc_test]

    
def muticlass_macro_f1(y_pred, y_true, data_type, step = "step1"):
    
    num_classes1 = len(np.unique(y_true))
    num_classes2 = len(np.unique(y_pred))
    # y_pred = y_pred2_test
    # y_true =  y_test
    classes = np.arange(num_classes1)
    
    conf_mat = np.zeros((3, 3), dtype=int)
    
    y_pred = convert_arr(y_pred)
    y_true = convert_arr(y_true)
    
    for t, p in zip(y_true.astype(np.int_), y_pred.astype(np.int_)):
        conf_mat[t, p] += 1
    
    # conf_mat[i, j] = number of samples with true class i and predicted class j
    print(f"confmat for {data_type}: \n", conf_mat)
    
    f1_scores = []
    precision_L = []
    recall_L = []
    # TP_list = []
    # FP_list = []
    # FN_list = []
    for c in classes:
    # For class c:
    # True Positives (TP): conf_mat[c, c]
    # False Positives (FP): sum of predictions as c but true != c
    # False Negatives (FN): sum of true c but predicted != c
        TP = conf_mat[c, c]
        FP = conf_mat[:, c].sum() - TP
        FN = conf_mat[c, :].sum() - TP
        print("TP,FP,FN:", [TP, FP, FN])
    
        # # Precision and Recall
        precision = TP/ (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
        precision_L.append(precision)
        recall_L.append(recall)
       
        if precision == 0.0 or recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
    
        f1_scores.append(f1)
    
    # F1 Score
    
    # Macro F1: average F1 over all classes
    macro_f1 = np.mean(f1_scores)
    macro_precision = np.mean(precision_L)
    macro_recall = np.mean(recall_L)

    print(f"step: {step}, data type: {data_type} set")
    print(f"Macro precision for {step}, {data_type} set: ", macro_precision)
    print(f"Macro recall Score for {step}, {data_type} set: ", macro_recall)
    print(f"Macro F1 Score for {step}, {data_type} set: ", macro_f1)
    return([macro_precision, macro_recall, macro_f1])




def calculate_real_mse(actual_times, predicted_times, actual_events, predicted_events):
    """
    Calculate the real Mean Squared Error (MSE) for survival time predictions,
    with special handling for cases where both actual and predicted events are 0.
    
    Parameters:
    - actual_times (array-like): Actual survival times.
    - predicted_times (array-like): Predicted survival times.
    - actual_events (array-like): Actual event types (e.g., 'OTB', 'DBE', 'Censored').
    - predicted_events (array-like): Predicted event types (e.g., 'OTB', 'DBE', 'Censored').
    
    Returns:
    - MSE_real (float): Real MSE with special case handling.
    """
    actual_times = np.array(actual_times)
    predicted_times = np.array(predicted_times)
    actual_events = np.array(actual_events)
    predicted_events = np.array(predicted_events)
    
    # Calculate time differences
    time_diff = actual_times - predicted_times
    
    # Initialize penalties
    penalty = np.ones_like(actual_times, dtype=float)
    
    # Apply special case: if actual_events and predicted_events are both 0, set penalty to 0
    penalty[(actual_events == 0) & (predicted_events == 0)] = 0
    
    # Compute penalized squared error
    squared_error = penalty * (time_diff ** 2)
    
    # Calculate mean and round to 3 decimal places
    MSE_real = float(format(np.mean(squared_error), '.3f'))
    
    print("Real MSE:", MSE_real)
    return MSE_real

def calculate_penalized_mse(
    actual_times,
    predicted_times,
    actual_events,
    predicted_events,
    penalty=np.sqrt(2)
):
    """
    Calculate penalized MSE for survival prediction.
    """

    actual_times = np.array(actual_times)
    predicted_times = np.array(predicted_times)
    actual_events = np.array(actual_events)
    predicted_events = np.array(predicted_events)

    # Time differences
    time_diff = actual_times - predicted_times

    # Initialize weights
    penalty_weights = np.ones_like(actual_events, dtype=float)

    # Apply penalty for misclassification
    penalty_weights[(actual_events != predicted_events)] = penalty

    # Optional: zero penalty when both are censored
    penalty_weights[
        (actual_events == 0) & (predicted_events == 0)
    ] = 0

    # Penalized squared error
    squared_error = penalty_weights * (time_diff ** 2)

    MSE_adj = float(format(np.mean(squared_error), '.3f'))

    print("penalized_mse:", MSE_adj)

    return MSE_adj



def calculate_balanced_penalized_mse(actual_times, predicted_times, actual_events, predicted_events, min_penalty=min_penalty_val, max_penalty=max_penalty_val):
    """
    Calculate penalized MSE incorporating balanced accuracy principles to address class imbalance,
    with configurable minimum and maximum off-diagonal penalties.
    
    Parameters:
    - actual_times (array-like): Actual survival times
    - predicted_times (array-like): Predicted survival times
    - actual_events (array-like): Actual event types (e.g., 0=censored, 1=OTB, 2=DBE)
    - predicted_events (array-like): Predicted event types
    - min_penalty (float): Minimum penalty for misclassifications (default: 1.2)
    - max_penalty (float): Maximum penalty for misclassifications (default: 5.0)
    
    Returns:
    - balanced_penalized_mse (float): Class-balanced penalized MSE with configured penalty range
    """
    actual_times = np.array(actual_times)
    predicted_times = np.array(predicted_times)
    actual_events = np.array(actual_events)
    predicted_events = np.array(predicted_events)
    
    # Calculate time differences
    time_diff = actual_times - predicted_times
    squared_error = time_diff ** 2
    
    # Get unique classes and their counts
    unique_classes = np.unique(actual_events)
    class_counts = {cls: np.sum(actual_events == cls) for cls in unique_classes}
    num_samples = len(actual_events)
    
    # Calculate inverse class frequencies (to give more weight to minority classes)
    class_weights = {cls: num_samples / (len(unique_classes) * count) for cls, count in class_counts.items()}
    
    # Scale weights to fit within the min_penalty and max_penalty range
    min_weight = min(class_weights.values())
    max_weight = max(class_weights.values())
    
    # Linear scaling of weights to the desired penalty range
    adjusted_weights = {}
    for cls, weight in class_weights.items():
        if max_weight == min_weight:  # Handle the case where all classes have the same count
            adjusted_weights[cls] = min_penalty
        else:
            # Scale to [min_penalty, max_penalty]
            norm_weight = (weight - min_weight) / (max_weight - min_weight)  # Normalize to [0,1]
            adjusted_weights[cls] = min_penalty + norm_weight * (max_penalty - min_penalty)
    
    # Apply weights to penalize misclassifications based on balanced accuracy principles
    penalty = np.ones_like(actual_events, dtype=float)
    
    # Apply class weights to correct and incorrect classifications
    for true_cls in unique_classes:
        # Correct classifications
        is_correct = (actual_events == true_cls) & (predicted_events == true_cls)
        penalty[is_correct] = 1.0  # Base penalty for correct classification
        
        # Incorrect classifications (higher penalty, adjusted by class weight)
        for pred_cls in unique_classes:
            if pred_cls != true_cls:
                is_incorrect = (actual_events == true_cls) & (predicted_events == pred_cls)
                # Apply scaled penalty for misclassification
                penalty[is_incorrect] = adjusted_weights[true_cls]
    
    # Special case: No penalty for correct censoring prediction
    penalty[(actual_events == 0) & (predicted_events == 0)] = 0
                
    # Compute penalized squared error
    balanced_squared_error = penalty * squared_error
    
    # Calculate mean with 3 decimal precision
    balanced_mse_adj = float(format(np.mean(balanced_squared_error), '.3f'))
    
    return balanced_mse_adj

def calculate_f1_penalized_mse(actual_times, predicted_times, actual_events, predicted_events, min_penalty=min_penalty_val, max_penalty=max_penalty_val):
    """
    Calculate penalized MSE incorporating F1 score principles to address class imbalance,
    with configurable minimum and maximum off-diagonal penalties.
    
    Parameters:
    - actual_times (array-like): Actual survival times
    - predicted_times (array-like): Predicted survival times
    - actual_events (array-like): Actual event types (e.g., 0=censored, 1=OTB, 2=DBE)
    - predicted_events (array-like): Predicted event types
    - min_penalty (float): Minimum penalty for misclassifications (default: 1.2)
    - max_penalty (float): Maximum penalty for misclassifications (default: 5.0)
    
    Returns:
    - f1_penalized_mse (float): F1-weighted penalized MSE with configured penalty range
    """
    actual_times = np.array(actual_times)
    predicted_times = np.array(predicted_times)
    actual_events = np.array(actual_events)
    predicted_events = np.array(predicted_events)
    
    # Calculate time differences
    time_diff = actual_times - predicted_times
    squared_error = time_diff ** 2
    
    # Get unique classes
    unique_classes = np.unique(actual_events)
    
    # Initialize penalty array
    penalty = np.ones_like(actual_events, dtype=float)
    
    # Calculate class-specific metrics and penalties
    f1_weights = {}
    for cls in unique_classes:
        # True positives, false positives, false negatives for this class
        true_positives = np.sum((actual_events == cls) & (predicted_events == cls))
        false_positives = np.sum((actual_events != cls) & (predicted_events == cls))
        false_negatives = np.sum((actual_events == cls) & (predicted_events != cls))
        
        # Precision and recall (with smoothing to avoid division by zero)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        
        # F1 score (harmonic mean of precision and recall)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
        
        # Calculate inverse F1 weight (higher weight for classes with lower F1)
        # Store for scaling later
        f1_weights[cls] = 1 / max(f1, 1e-8)
    
    # Scale the f1_weights to fit within the specified penalty range
    min_f1_weight = min(f1_weights.values())
    max_f1_weight = max(f1_weights.values())
    
    # Apply penalties based on scaled F1 weights
    for cls in unique_classes:
        # Correct classifications
        penalty[(actual_events == cls) & (predicted_events == cls)] = 1.0
        
        # Scale the weight to the desired range
        if max_f1_weight == min_f1_weight:  # Handle case where all F1 scores are the same
            scaled_weight = min_penalty
        else:
            # Normalize to [0,1] then scale to [min_penalty, max_penalty]
            norm_weight = (f1_weights[cls] - min_f1_weight) / (max_f1_weight - min_f1_weight)
            scaled_weight = min_penalty + norm_weight * (max_penalty - min_penalty)
        
        # Incorrect classifications with scaled penalties
        for pred_cls in unique_classes:
            if pred_cls != cls:
                is_incorrect = (actual_events == cls) & (predicted_events == pred_cls)
                penalty[is_incorrect] = scaled_weight
    
    # Special case: No penalty for correct censoring prediction (class 0)
    penalty[(actual_events == 0) & (predicted_events == 0)] = 0
    
    # Compute penalized squared error
    f1_penalized_squared_error = penalty * squared_error
    
    # Calculate mean with 3 decimal precision
    f1_penalized_mse = float(format(np.mean(f1_penalized_squared_error), '.3f'))
    
    return f1_penalized_mse

def calculate_recall_penalized_mse(actual_times, predicted_times, actual_events, predicted_events, min_penalty=min_penalty_val, max_penalty=max_penalty_val):
    """
    Calculate penalized MSE incorporating recall principles to address class imbalance,
    with configurable minimum and maximum off-diagonal penalties.
    
    Parameters:
    - actual_times (array-like): Actual survival times
    - predicted_times (array-like): Predicted survival times
    - actual_events (array-like): Actual event types (e.g., 0=censored, 1=OTB, 2=DBE)
    - predicted_events (array-like): Predicted event types
    - min_penalty (float): Minimum penalty for misclassifications (default: 1.2)
    - max_penalty (float): Maximum penalty for misclassifications (default: 5.0)
    
    Returns:
    - recall_penalized_mse (float): Recall-weighted penalized MSE with configured penalty range
    """
    actual_times = np.array(actual_times)
    predicted_times = np.array(predicted_times)
    actual_events = np.array(actual_events)
    predicted_events = np.array(predicted_events)
    
    # Calculate time differences
    time_diff = actual_times - predicted_times
    squared_error = time_diff ** 2
    
    # Get unique classes
    unique_classes = np.unique(actual_events)
    
    # Initialize penalty array and recall weights dictionary
    penalty = np.ones_like(actual_events, dtype=float)
    recall_weights = {}
    
    # Calculate class-specific metrics and store weights
    for cls in unique_classes:
        # Calculate recall for this class
        true_positives = np.sum((actual_events == cls) & (predicted_events == cls))
        actual_class_count = np.sum(actual_events == cls)
        
        # Recall with smoothing to avoid division by zero
        recall = true_positives / max(actual_class_count, 1)
        
        # Inverse recall weight (higher penalty for classes with lower recall)
        recall_weights[cls] = 1 / max(recall, 1e-8)
    
    # Scale the recall_weights to fit within the specified penalty range
    min_recall_weight = min(recall_weights.values())
    max_recall_weight = max(recall_weights.values())
    
    # Apply penalties based on scaled recall weights
    for cls in unique_classes:
        # Correct classifications
        penalty[(actual_events == cls) & (predicted_events == cls)] = 1.0
        
        # Scale the weight to the desired range
        if max_recall_weight == min_recall_weight:  # Handle case where all recall values are the same
            scaled_weight = min_penalty
        else:
            # Normalize to [0,1] then scale to [min_penalty, max_penalty]
            norm_weight = (recall_weights[cls] - min_recall_weight) / (max_recall_weight - min_recall_weight)
            scaled_weight = min_penalty + norm_weight * (max_penalty - min_penalty)
        
        # Incorrect classifications with scaled penalties
        for pred_cls in unique_classes:
            if pred_cls != cls:
                is_incorrect = (actual_events == cls) & (predicted_events == pred_cls)
                penalty[is_incorrect] = scaled_weight
    
    # Special case: No penalty for correct censoring prediction (class 0)
    penalty[(actual_events == 0) & (predicted_events == 0)] = 0
    
    # Compute penalized squared error
    recall_penalized_squared_error = penalty * squared_error
    
    # Calculate mean with 3 decimal precision
    recall_penalized_mse = float(format(np.mean(recall_penalized_squared_error), '.3f'))
    
    return recall_penalized_mse
    

def calculate_dual_weighted_penalized_mse(actual_times, predicted_times, actual_events, predicted_events, 
                                                 balance_weight=0.5, f1_weight=0.5,
                                                 min_penalty=min_penalty_val, max_penalty=max_penalty_val):
    """
    Calculate Dual-Weighted Penalized MSE (DW-PMSE) with bounded penalties.
    
    Parameters:
    - actual_times (array-like): Actual survival times
    - predicted_times (array-like): Predicted survival times
    - actual_events (array-like): Actual event types (e.g., 0=censored, 1=OTB, 2=DBE)
    - predicted_events (array-like): Predicted event types
    - balance_weight (float): Weight for the class-balanced component (default: 0.5)
    - f1_weight (float): Weight for the F1-based component (default: 0.5)
    - min_penalty (float): Minimum penalty value for misclassifications (default: 1.2)
    - max_penalty (float): Maximum penalty value for misclassifications (default: 2.0)
    
    Returns:
    - dw_penalized_mse (float): Bounded Dual-Weighted Penalized MSE
    """
    import numpy as np
    
    actual_times = np.array(actual_times)
    predicted_times = np.array(predicted_times)
    actual_events = np.array(actual_events)
    predicted_events = np.array(predicted_events)
    
    # Calculate time differences
    time_diff = actual_times - predicted_times
    squared_error = time_diff ** 2
    
    # Get unique classes and their counts
    unique_classes = np.unique(actual_events)
    class_counts = {cls: np.sum(actual_events == cls) for cls in unique_classes}
    num_samples = len(actual_events)
    
    # Calculate inverse class frequencies (balanced component)
    class_weights = {cls: num_samples / (len(unique_classes) * count) for cls, count in class_counts.items()}
    
    # Initialize penalty array
    penalty = np.ones_like(actual_events, dtype=float)
    
    # Calculate class-specific F1 metrics
    f1_weights = {}
    for cls in unique_classes:
        # True positives, false positives, false negatives for this class
        true_positives = np.sum((actual_events == cls) & (predicted_events == cls))
        false_positives = np.sum((actual_events != cls) & (predicted_events == cls))
        false_negatives = np.sum((actual_events == cls) & (predicted_events != cls))
        
        # Precision and recall (with smoothing to avoid division by zero)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        
        # F1 score (harmonic mean of precision and recall)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
        
        # Calculate inverse F1 weight (capped at 5.0)
        f1_weights[cls] = min(5.0, np.log(1 / max(f1, 1e-8)))
    
    # Calculate raw penalties for each class
    raw_penalties = {}
    for cls in unique_classes:
        balance_penalty = np.sqrt(2) * class_weights[cls]
        f1_penalty = np.sqrt(2) * f1_weights[cls]
        raw_penalties[cls] = (balance_weight * balance_penalty) + (f1_weight * f1_penalty)
    
    # Find min and max raw penalties (excluding class 0)
    non_zero_penalties = [p for c, p in raw_penalties.items() if c != 0]
    if non_zero_penalties:
        min_raw = min(non_zero_penalties)
        max_raw = max(non_zero_penalties)
        
        # Create scaling function to map from [min_raw, max_raw] to [min_penalty, max_penalty]
        def scale_penalty(p):
            if max_raw == min_raw:  # Avoid division by zero
                return min_penalty
            return min_penalty + (max_penalty - min_penalty) * (p - min_raw) / (max_raw - min_raw)
    else:
        # If there are no non-zero classes, use default scaling
        def scale_penalty(p):
            return min_penalty
    
    # Apply bounded dual-weighted penalties
    for true_cls in unique_classes:
        # Correct classifications
        is_correct = (actual_events == true_cls) & (predicted_events == true_cls)
        penalty[is_correct] = 1.0
        
        # Incorrect classifications
        for pred_cls in unique_classes:
            if pred_cls != true_cls:
                is_incorrect = (actual_events == true_cls) & (predicted_events == pred_cls)
                
                # Scale the penalty to be within bounds
                if true_cls != 0:  # Skip scaling for class 0
                    scaled_penalty = scale_penalty(raw_penalties[true_cls])
                    penalty[is_incorrect] = scaled_penalty
                else:
                    # For class 0, use the minimum penalty
                    penalty[is_incorrect] = min_penalty
    
    # Special case: No penalty for correct censoring prediction
    penalty[(actual_events == 0) & (predicted_events == 0)] = 0
    
    # Compute bounded dual-weighted penalized squared error
    bounded_dw_penalized_squared_error = penalty * squared_error
    
    # Calculate mean with 3 decimal precision
    bounded_dw_penalized_mse = float(format(np.mean(bounded_dw_penalized_squared_error), '.3f'))
    
    return bounded_dw_penalized_mse
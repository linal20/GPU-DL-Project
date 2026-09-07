from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
from sklearn.metrics import (make_scorer, mean_squared_error, accuracy_score, 
                            f1_score, recall_score, precision_score, 
                            balanced_accuracy_score, matthews_corrcoef,
                            confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

import importlib
import metrics_functions as mt
importlib.reload(mt)



def cross_validate_models_with_balanced_mse_wtrain(models, X_reg, y_reg, X_clf=None, y_clf=None, n_folds=5, n_jobs=-1, resampling_strategy="smote"):
    """
    Perform cross-validation on multiple regression and classification models,
    with balanced penalized MSE calculation and resampling for imbalanced datasets.
    Evaluates and reports metrics for both test and training sets.
    
    Parameters:
    -----------
    models : list of tuples
        Each tuple contains (model_name, regressor, classifier)
    X_reg : array-like
        Features for regression tasks
    y_reg : array-like
        Target for regression tasks
    X_clf : array-like, optional
        Features for classification tasks (if None, only regression is evaluated)
    y_clf : array-like, optional
        Target for classification tasks (if None, only regression is evaluated)
    n_folds : int, default=5
        Number of cross-validation folds
    n_jobs : int, default=3
        Number of parallel jobs for cross-validation
    resampling_strategy : str, default="smote"
        Resampling strategy to use: "smote", "adasyn", "random_over", "random_under", or "none"
    
    Returns:
    --------
    tuple
        (cv_results, cv_results_train, confusion_matrices, confusion_matrices_train, fold_predictions, 
         balanced_penalized_mse_results, f1_penalized_mse_results, recall_penalized_mse_results, 
         penalized_mse_results, real_mse_results)
    """

    min_penalty = 1.2
    max_penalty = 3

    # Set up cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

 


    # Dictionary to store results
    cv_results = {'train':{}, 'test':{}}
    confusion_matrices = {'train':{}, 'test':{}}
    fold_predictions = {}
    balanced_penalized_mse_results = {}
    f1_penalized_mse_results = {}
    recall_penalized_mse_results = {}
    penalized_mse_results = {}
    # dynamic_penalized_mse_results = {}
    # dynamic_f1_penalized_mse_results = {}
    real_mse_results = {} 
    dual_weighted_penalized_mse_results = {}
    

    # Import resampling methods
    if resampling_strategy.lower() != "none":
        try:
            from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler
            from collections import Counter
            print(f"  Using resampling strategy: {resampling_strategy}")
        except ImportError:
            print("  Warning: imbalanced-learn package not found. Install with: pip install imbalanced-learn")
            print("  Proceeding without resampling...")
            resampling_strategy = "none"

    # Check if y_clf needs encoding (if it's an object array or contains non-numeric values)
    label_encoder = None
    if X_clf is not None and y_clf is not None:
        if y_clf.dtype == 'object' or not np.issubdtype(y_clf.dtype, np.number):
            print("  Detected non-numeric labels. Using label encoding...")
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_clf_encoded = label_encoder.fit_transform(y_clf)
            # Create a mapping for reference
            label_mapping = {label: encoded for label, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_)))}
            print(f"  Label mapping: {label_mapping}")
            # Replace original with encoded
            y_clf = y_clf_encoded
    print(models)
    # Iterate through models
    for model_name, reg_model, clf_model in models:
        print(" ")
        print(" ")
        print(f"\nEvaluating {model_name}...")
        model_results = {'train':{},'test':{}
                            }
        fold_preds = {
                'regression': {'train': [], 'test': []},
                'classification': {'train': [], 'test': []}
            }
            
        # Regression cross-validation
        try:
            print(f"  Running regression cross-validation...")
            # Initialize arrays to store predictions for each fold
            reg_test_fold_predictions = []
            reg_train_fold_predictions = []
            
            # Arrays to store metrics
            test_mse_scores = []
            test_r2_scores = []
            train_mse_scores = []
            train_r2_scores = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_reg)):
                # Handle both pandas DataFrame/Series and numpy arrays
                if hasattr(X_reg, 'iloc'):
                    X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
                else:
                    X_train, X_test = X_reg[train_idx], X_reg[test_idx]
                    
                if hasattr(y_reg, 'iloc'):
                    y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
                else:
                    y_train, y_test = y_reg[train_idx], y_reg[test_idx]
                
                # Train and predict
                reg_model.fit(X_train, y_train)
                
                # Predict on test set
                y_test_pred = reg_model.predict(X_test)
                
                # Predict on train set
                y_train_pred = reg_model.predict(X_train)
                
                # Calculate test set metrics
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_r2 = reg_model.score(X_test, y_test)  # R^2 score
                
                # Calculate train set metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_r2 = reg_model.score(X_train, y_train)  # R^2 score
                
                # Store scores
                test_mse_scores.append(test_mse)
                test_r2_scores.append(test_r2)
                train_mse_scores.append(train_mse)
                train_r2_scores.append(train_r2)
                
                # Store test predictions with corresponding indices and true values
                reg_test_fold_predictions.append({
                    'fold': fold_idx,
                    'indices': test_idx,
                    'y_true': y_test.values.copy() if hasattr(y_test, 'values') else y_test.copy(),
                    'y_pred': y_test_pred.copy() if isinstance(y_test_pred, np.ndarray) else np.array(y_test_pred)
                })
                
                # Store train predictions with corresponding indices and true values
                reg_train_fold_predictions.append({
                    'fold': fold_idx,
                    'indices': train_idx,
                    'y_true': y_train.values.copy() if hasattr(y_train, 'values') else y_train.copy(),
                    'y_pred': y_train_pred.copy() if isinstance(y_train_pred, np.ndarray) else np.array(y_train_pred)
                })
                
                # print(f"    Fold {fold_idx+1}:")
                # print(f"      Test  - MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
                # print(f"      Train - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
            
            # Store predictions
            fold_preds['regression']['test'] = reg_test_fold_predictions
            fold_preds['regression']['train'] = reg_train_fold_predictions
            
            # Store test metrics
            model_results['test']['reg_mse_scores'] = np.array(test_mse_scores)
            model_results['test']['reg_mse_mean'] = np.mean(test_mse_scores)
            model_results['test']['reg_mse_std'] = np.std(test_mse_scores)
            
            model_results['test']['reg_r2_scores'] = np.array(test_r2_scores)
            model_results['test']['reg_r2_mean'] = np.mean(test_r2_scores)
            model_results['test']['reg_r2_std'] = np.std(test_r2_scores)
            
            # Store train metrics
            model_results['train']['reg_mse_scores'] = np.array(train_mse_scores)
            model_results['train']['reg_mse_mean'] = np.mean(train_mse_scores)
            model_results['train']['reg_mse_std'] = np.std(train_mse_scores)
            
            model_results['train']['reg_r2_scores'] = np.array(train_r2_scores)
            model_results['train']['reg_r2_mean'] = np.mean(train_r2_scores)
            model_results['train']['reg_r2_std'] = np.std(train_r2_scores)
            
            # print(f"  Regression Test MSE: {model_results['test']['reg_mse_mean']:.4f} ± {model_results['test']['reg_mse_std']:.4f}")
            # print(f"  Regression Test R²: {model_results['test']['reg_r2_mean']:.4f} ± {model_results['test']['reg_r2_std']:.4f}")
            # print(f"  Regression Train MSE: {model_results['train']['reg_mse_mean']:.4f} ± {model_results['train']['reg_mse_std']:.4f}")
            # print(f"  Regression Train R²: {model_results['train']['reg_r2_mean']:.4f} ± {model_results['train']['reg_r2_std']:.4f}")
            
        except Exception as e:
            print(f"  Regression error: {str(e)}")
            model_results['reg_error'] = str(e)
           

        try:
            print(f"  Running classification cross-validation...")
            # Initialize arrays to store predictions for each fold
            clf_test_fold_predictions = []
            clf_train_fold_predictions = []
            
            # Dictionary to store confusion matrices for each fold
            test_fold_confusion_matrices = {}
            train_fold_confusion_matrices = {}
            
            # Manual k-fold cross validation
            test_acc_scores = []
            test_f1_macro_scores = []
            test_bal_acc_scores = []
            test_recall_scores = []
            test_precision_scores = []
            test_mcc_scores = []
            
            train_acc_scores = []
            train_f1_macro_scores = []
            train_bal_acc_scores = []
            train_recall_scores = []
            train_precision_scores = []
            train_mcc_scores = []
            
            # Accumulate predictions for overall confusion matrix
            all_test_y_true = []
            all_test_y_pred = []
            all_train_y_true = []
            all_train_y_pred = []
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_clf)):
                # Handle both pandas DataFrame/Series and numpy arrays
                if hasattr(X_clf, 'iloc'):
                    X_train, X_test = X_clf.iloc[train_idx], X_clf.iloc[test_idx]
                else:
                    X_train, X_test = X_clf[train_idx], X_clf[test_idx]
                    
                if hasattr(y_clf, 'iloc'):
                    y_train, y_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]
                else:
                    y_train, y_test = y_clf[train_idx], y_clf[test_idx]
                
                # Apply resampling to the training data
                X_original_train = X_train.copy()
                y_original_train = y_train.copy()
                
                if resampling_strategy.lower() != "none" and X_clf is not None and y_clf is not None:
                    # Check class distribution before resampling
                    if hasattr(y_train, 'values'):
                        before_counts = Counter(y_train.values)
                    else:
                        before_counts = Counter(y_train)
                    print(f"    Before resampling - Class distribution: {dict(before_counts)}")
                    
                    # Apply the selected resampling strategy
                    try:
                        X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
                        y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
                        
                        if resampling_strategy.lower() == "smote":
                            resampler = SMOTE(random_state=42)
                        elif resampling_strategy.lower() == "adasyn":
                            resampler = ADASYN(random_state=42)
                        elif resampling_strategy.lower() == "random_over":
                            resampler = RandomOverSampler(random_state=42)
                        elif resampling_strategy.lower() == "random_under":
                            resampler = RandomUnderSampler(random_state=42)
                        
                        X_resampled, y_resampled = resampler.fit_resample(X_train_array, y_train_array)
                        
                        # Convert back to original type if needed
                        if hasattr(X_train, 'iloc'):
                            import pandas as pd
                            if hasattr(X_train, 'columns'):
                                X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
                            else:
                                X_train = pd.DataFrame(X_resampled)
                        else:
                            X_train = X_resampled
                            
                        if hasattr(y_train, 'iloc'):
                            y_train = pd.Series(y_resampled)
                        else:
                            y_train = y_resampled
                        
                        # Check class distribution after resampling
                        after_counts = Counter(y_resampled)
                        print(f"    After resampling - Class distribution: {dict(after_counts)}")
                    except Exception as e:
                        print(f"    Resampling error in fold {fold_idx+1}: {str(e)}")
                        print(f"    Proceeding with original imbalanced data...")
                
                                # Train and predict
                clf_model.fit(X_train, y_train)
                y_test_pred = clf_model.predict(X_test)
                y_train_pred = clf_model.predict(X_original_train)
                
                # Generate confusion matrix for test fold
                from sklearn.metrics import confusion_matrix
                test_fold_cm = confusion_matrix(y_test, y_test_pred)
                test_fold_confusion_matrices[fold_idx] = test_fold_cm
                
                # Generate confusion matrix for train fold
                train_fold_cm = confusion_matrix(y_original_train, y_train_pred)
                train_fold_confusion_matrices[fold_idx] = train_fold_cm
                
                # Calculate test metrics
                test_acc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred, average='macro')
                test_bal_acc = balanced_accuracy_score(y_test, y_test_pred)
                test_recall = recall_score(y_test, y_test_pred, average='macro')
                test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
                test_mcc = matthews_corrcoef(y_test, y_test_pred)
                
                # Calculate train metrics
                train_acc = accuracy_score(y_original_train, y_train_pred)
                train_f1 = f1_score(y_original_train, y_train_pred, average='macro')
                train_bal_acc = balanced_accuracy_score(y_original_train, y_train_pred)
                train_recall = recall_score(y_original_train, y_train_pred, average='macro')
                train_precision = precision_score(y_original_train, y_train_pred, average='macro', zero_division=0)
                train_mcc = matthews_corrcoef(y_original_train, y_train_pred)
                
                # Store scores
                test_acc_scores.append(test_acc)
                test_f1_macro_scores.append(test_f1)
                test_bal_acc_scores.append(test_bal_acc)
                test_recall_scores.append(test_recall)
                test_precision_scores.append(test_precision)
                test_mcc_scores.append(test_mcc)
                
                train_acc_scores.append(train_acc)
                train_f1_macro_scores.append(train_f1)
                train_bal_acc_scores.append(train_bal_acc)
                train_recall_scores.append(train_recall)
                train_precision_scores.append(train_precision)
                train_mcc_scores.append(train_mcc)
                
                # Store test predictions
                clf_test_fold_predictions.append({
                    'fold': fold_idx,
                    'indices': test_idx,
                    'y_true': y_test.values.copy() if hasattr(y_test, 'values') else y_test.copy(),
                    'y_pred': y_test_pred.copy() if isinstance(y_test_pred, np.ndarray) else np.array(y_test_pred),
                    'confusion_matrix': test_fold_cm
                })
                
                # Store train predictions
                clf_train_fold_predictions.append({
                    'fold': fold_idx,
                    'indices': train_idx,
                    'y_true': y_original_train.values.copy() if hasattr(y_original_train, 'values') else y_original_train.copy(),
                    'y_pred': y_train_pred.copy() if isinstance(y_train_pred, np.ndarray) else np.array(y_train_pred),
                    'confusion_matrix': train_fold_cm
                })
                
                # Accumulate for overall confusion matrix
                all_test_y_true.extend(y_test.values if hasattr(y_test, 'values') else y_test)
                all_test_y_pred.extend(y_test_pred)
                all_train_y_true.extend(y_original_train.values if hasattr(y_original_train, 'values') else y_original_train)
                all_train_y_pred.extend(y_train_pred)
                
                print(f"    Fold {fold_idx+1}:")
                print(f"      Test  - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Balanced Accuracy: {test_bal_acc:.4f}")
                print(f"      Train - Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, Balanced Accuracy: {train_bal_acc:.4f}")
            
            # Store predictions
            fold_preds['classification']['test'] = clf_test_fold_predictions
            fold_preds['classification']['train'] = clf_train_fold_predictions

        except Exception as e:
            print(f"  Classification error: {str(e)}")
            model_results['clf_error'] = str(e)   
        
        # Calculate balanced penalized MSE across folds for test data
        test_balanced_penalized_mse_scores = []
        test_f1_penalized_mse_scores = []
        test_recall_penalized_mse_scores = []
        test_penalized_mse_scores = []
        test_real_mse_scores = []  
        # test_dynamic_penalized_mse_scores = []
        test_dynamic_f1_penalized_mse_scores = []
        # Calculate balanced penalized MSE across folds for train data
        train_balanced_penalized_mse_scores = []
        train_f1_penalized_mse_scores = []
        train_recall_penalized_mse_scores = []
        train_penalized_mse_scores = []
        train_real_mse_scores = []  
        train_dynamic_penalized_mse_scores = []
        train_dynamic_f1_penalized_mse_scores = []
        
        test_dual_weighted_penalized_mse_scores = []
        train_dual_weighted_penalized_mse_scores = []
        # Process test data metrics
        for reg_fold, clf_fold in zip(reg_test_fold_predictions, clf_test_fold_predictions):
            # Verify the indices match (they should, but let's be safe)
            if np.array_equal(reg_fold['indices'], clf_fold['indices']):
                # Calculate standard penalized MSE
                clf_fold['y_pred']=clf_fold['y_pred'].flatten()
                pen_mse = mt.calculate_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred']
                )
                test_penalized_mse_scores.append(pen_mse)
                # Calculate real MSE
                real_mse = mt.calculate_real_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred']
                )
                test_real_mse_scores.append(real_mse)
                
                # Calculate balanced penalized MSE
                bal_pen_mse = mt.calculate_balanced_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                test_balanced_penalized_mse_scores.append(bal_pen_mse)

                # dynamic_mse = mt.calculate_dynamic_penalized_mse(
                #     actual_times=reg_fold['y_true'],
                #     predicted_times=reg_fold['y_pred'],
                #     actual_events=clf_fold['y_true'],
                #     predicted_events=clf_fold['y_pred']
                # )
                # test_dynamic_penalized_mse_scores.append(dynamic_mse)
                
                f1_pen_mse = mt.calculate_f1_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                test_f1_penalized_mse_scores.append(f1_pen_mse)

          
                recall_pen_mse = mt.calculate_recall_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                test_recall_penalized_mse_scores.append(recall_pen_mse)
           
                dual_weighted_pen_mse = mt.calculate_dual_weighted_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                test_dual_weighted_penalized_mse_scores.append(dual_weighted_pen_mse)
            
            else:
                print(f"    Warning: Test indices mismatch in fold {reg_fold['fold']+1}")


            
        # Process train data metrics
        for reg_fold, clf_fold in zip(reg_train_fold_predictions, clf_train_fold_predictions):
            # Verify the indices match
            clf_fold['y_pred'] = clf_fold['y_pred'].flatten()
            if np.array_equal(reg_fold['indices'], clf_fold['indices']):
                # Calculate real MSE
                real_mse = mt.calculate_real_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred']
                )
                train_real_mse_scores.append(real_mse)
                
                # Calculate balanced penalized MSE
                bal_pen_mse = mt.calculate_balanced_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                train_balanced_penalized_mse_scores.append(bal_pen_mse)

                # Calculate real MSE
                # dynamic_mse = mt.calculate_dynamic_penalized_mse(
                #     actual_times=reg_fold['y_true'],
                #     predicted_times=reg_fold['y_pred'],
                #     actual_events=clf_fold['y_true'],
                #     predicted_events=clf_fold['y_pred']
                # )
                # train_dynamic_penalized_mse_scores.append(dynamic_mse)
                
                f1_pen_mse = mt.calculate_f1_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                train_f1_penalized_mse_scores.append(f1_pen_mse)

                recall_pen_mse = mt.calculate_recall_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                train_recall_penalized_mse_scores.append(recall_pen_mse)
                
                # Calculate standard penalized MSE
                pen_mse = mt.calculate_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred']
                )
                train_penalized_mse_scores.append(pen_mse)

                
                # Add this to your train data processing loop
                dual_weighted_pen_mse = mt.calculate_dual_weighted_penalized_mse(
                    actual_times=reg_fold['y_true'],
                    predicted_times=reg_fold['y_pred'],
                    actual_events=clf_fold['y_true'],
                    predicted_events=clf_fold['y_pred'],
                    min_penalty = min_penalty,
                    max_penalty = max_penalty
                )
                train_dual_weighted_penalized_mse_scores.append(dual_weighted_pen_mse)
                
            else:
                print(f"    Warning: Train indices mismatch in fold {reg_fold['fold']+1}")


       
            # Store test real MSE results
            if test_real_mse_scores:
                real_mse_mean = np.mean(test_real_mse_scores)
                real_mse_std = np.std(test_real_mse_scores)
                model_results['test']['real_mse_scores'] = np.array(test_real_mse_scores)
                model_results['test']['real_mse_mean'] = real_mse_mean
                model_results['test']['real_mse_std'] = real_mse_std
                print(f"  Test Real MSE: {real_mse_mean:.4f} ± {real_mse_std:.4f}")
      

             # Store train real MSE results
            if train_real_mse_scores:
                real_mse_mean = np.mean(train_real_mse_scores)
                real_mse_std = np.std(train_real_mse_scores)
                model_results['train']['real_mse_scores'] = np.array(train_real_mse_scores)
                model_results['train']['real_mse_mean'] = real_mse_mean
                model_results['train']['real_mse_std'] = real_mse_std
                print(f"  Train Real MSE: {real_mse_mean:.4f} ± {real_mse_std:.4f}")
                real_mse_results[model_name] = {
                    'test': test_real_mse_scores,
                    'train': train_real_mse_scores
                }
            # Store test penalized MSE results
            if test_penalized_mse_scores:
                pen_mse_mean = np.mean(test_penalized_mse_scores)
                pen_mse_std = np.std(test_penalized_mse_scores)
                model_results['test']['penalized_mse_scores'] = np.array(test_penalized_mse_scores)
                model_results['test']['penalized_mse_mean'] = pen_mse_mean
                model_results['test']['penalized_mse_std'] = pen_mse_std
                print(f"  Test Penalized MSE: {pen_mse_mean:.4f} ± {pen_mse_std:.4f}")
                
            # Store train penalized MSE results
            if train_penalized_mse_scores:
                pen_mse_mean = np.mean(train_penalized_mse_scores)
                pen_mse_std = np.std(train_penalized_mse_scores)
                model_results['train']['penalized_mse_scores'] = np.array(train_penalized_mse_scores)
                model_results['train']['penalized_mse_mean'] = pen_mse_mean
                model_results['train']['penalized_mse_std'] = pen_mse_std
                print(f"  Train Penalized MSE: {pen_mse_mean:.4f} ± {pen_mse_std:.4f}")

                penalized_mse_results[model_name] = {
                    'test': test_penalized_mse_scores,
                    'train': train_penalized_mse_scores
                }

            
            # Store test balanced penalized MSE results
            if test_balanced_penalized_mse_scores:
                bal_pen_mse_mean = np.mean(test_balanced_penalized_mse_scores)
                bal_pen_mse_std = np.std(test_balanced_penalized_mse_scores)
                model_results['test']['balanced_penalized_mse_scores'] = np.array(test_balanced_penalized_mse_scores)
                model_results['test']['balanced_penalized_mse_mean'] = bal_pen_mse_mean
                model_results['test']['balanced_penalized_mse_std'] = bal_pen_mse_std
                print(f"  Test Balanced Penalized MSE: {bal_pen_mse_mean:.4f} ± {bal_pen_mse_std:.4f}")

            
            # Store train balanced penalized MSE results
            if train_balanced_penalized_mse_scores:
                bal_pen_mse_mean = np.mean(train_balanced_penalized_mse_scores)
                bal_pen_mse_std = np.std(train_balanced_penalized_mse_scores)
                model_results['train']['balanced_penalized_mse_scores'] = np.array(train_balanced_penalized_mse_scores)
                model_results['train']['balanced_penalized_mse_mean'] = bal_pen_mse_mean
                model_results['train']['balanced_penalized_mse_std'] = bal_pen_mse_std
                print(f"  Train Balanced Penalized MSE: {bal_pen_mse_mean:.4f} ± {bal_pen_mse_std:.4f}")

                balanced_penalized_mse_results[model_name] = {
                    'test': test_balanced_penalized_mse_scores,
                    'train': train_balanced_penalized_mse_scores
                }
       

      
            
        
            # Store test F1 penalized MSE results
            if test_f1_penalized_mse_scores:
                f1_pen_mse_mean = np.mean(test_f1_penalized_mse_scores)
                f1_pen_mse_std = np.std(test_f1_penalized_mse_scores)
                model_results['test']['f1_penalized_mse_scores'] = np.array(test_f1_penalized_mse_scores)
                model_results['test']['f1_penalized_mse_mean'] = f1_pen_mse_mean
                model_results['test']['f1_penalized_mse_std'] = f1_pen_mse_std
                print(f"  Test F1 Penalized MSE: {f1_pen_mse_mean:.4f} ± {f1_pen_mse_std:.4f}")
                

            
            # Store train F1 penalized MSE results
            if train_f1_penalized_mse_scores:
                f1_pen_mse_mean = np.mean(train_f1_penalized_mse_scores)
                f1_pen_mse_std = np.std(train_f1_penalized_mse_scores)
                model_results['train']['f1_penalized_mse_scores'] = np.array(train_f1_penalized_mse_scores)
                model_results['train']['f1_penalized_mse_mean'] = f1_pen_mse_mean
                model_results['train']['f1_penalized_mse_std'] = f1_pen_mse_std
                print(f"  Train F1 Penalized MSE: {f1_pen_mse_mean:.4f} ± {f1_pen_mse_std:.4f}")

                f1_penalized_mse_results[model_name] = {
                    'test': test_f1_penalized_mse_scores,
                    'train': train_f1_penalized_mse_scores
                }
            # Store test recall penalized MSE results
            if test_recall_penalized_mse_scores:
                recall_pen_mse_mean = np.mean(test_recall_penalized_mse_scores)
                recall_pen_mse_std = np.std(test_recall_penalized_mse_scores)
                model_results['test']['recall_penalized_mse_scores'] = np.array(test_recall_penalized_mse_scores)
                model_results['test']['recall_penalized_mse_mean'] = recall_pen_mse_mean
                model_results['test']['recall_penalized_mse_std'] = recall_pen_mse_std
                print(f"  Test Recall Penalized MSE: {recall_pen_mse_mean:.4f} ± {recall_pen_mse_std:.4f}")

            
            # Store train recall penalized MSE results
            if train_recall_penalized_mse_scores:
                recall_pen_mse_mean = np.mean(train_recall_penalized_mse_scores)
                recall_pen_mse_std = np.std(train_recall_penalized_mse_scores)
                model_results['train']['recall_penalized_mse_scores'] = np.array(train_recall_penalized_mse_scores)
                model_results['train']['recall_penalized_mse_mean'] = recall_pen_mse_mean
                model_results['train']['recall_penalized_mse_std'] = recall_pen_mse_std
                print(f"  Train Recall Penalized MSE: {recall_pen_mse_mean:.4f} ± {recall_pen_mse_std:.4f}")

                recall_penalized_mse_results[model_name] = {
                    'test': test_recall_penalized_mse_scores,
                    'train': train_recall_penalized_mse_scores
                }

            # Add these to store test results
            if test_dual_weighted_penalized_mse_scores:
                dual_weighted_pen_mse_mean = np.mean(test_dual_weighted_penalized_mse_scores)
                dual_weighted_pen_mse_std = np.std(test_dual_weighted_penalized_mse_scores)
                model_results['test']['dual_weighted_penalized_mse_scores'] = np.array(test_dual_weighted_penalized_mse_scores)
                model_results['test']['dual_weighted_penalized_mse_mean'] = dual_weighted_pen_mse_mean
                model_results['test']['dual_weighted_penalized_mse_std'] = dual_weighted_pen_mse_std
                print(f"  Test Dual-Weighted Penalized MSE: {dual_weighted_pen_mse_mean:.4f} ± {dual_weighted_pen_mse_std:.4f}")

            
            # Add these to store train results
            if train_dual_weighted_penalized_mse_scores:
                dual_weighted_pen_mse_mean = np.mean(train_dual_weighted_penalized_mse_scores)
                dual_weighted_pen_mse_std = np.std(train_dual_weighted_penalized_mse_scores)
                model_results['train']['dual_weighted_penalized_mse_scores'] = np.array(train_dual_weighted_penalized_mse_scores)
                model_results['train']['dual_weighted_penalized_mse_mean'] = dual_weighted_pen_mse_mean
                model_results['train']['dual_weighted_penalized_mse_std'] = dual_weighted_pen_mse_std
                print(f"  Train Dual-Weighted Penalized MSE: {dual_weighted_pen_mse_mean:.4f} ± {dual_weighted_pen_mse_std:.4f}")
                
                dual_weighted_penalized_mse_results[model_name] = {
                    'test': test_dual_weighted_penalized_mse_scores,
                    'train': train_dual_weighted_penalized_mse_scores
                }
                
       # Store test classification metrics
        model_results['train']['clf_acc_scores'] = np.array(train_acc_scores)
        model_results['train']['clf_acc_mean'] = np.mean(train_acc_scores)
        model_results['train']['clf_acc_std'] = np.std(train_acc_scores)
        print(f"  Train Accuracy: {model_results['train']['clf_acc_mean']:.4f} ± {model_results['train']['clf_acc_std']:.4f}")

        model_results['train']['clf_f1_scores'] = np.array(train_f1_macro_scores )
        model_results['train']['clf_f1_mean'] = np.mean(train_f1_macro_scores )
        model_results['train']['clf_f1_std'] = np.std(train_f1_macro_scores )
        
        model_results['train']['clf_bal_acc_scores'] = np.array(train_bal_acc_scores)
        model_results['train']['clf_bal_acc_mean'] = np.mean(train_bal_acc_scores)
        model_results['train']['clf_bal_acc_std'] = np.std(train_bal_acc_scores)
        print(f"  Train Balanced Accuracy: {model_results['train']['clf_bal_acc_mean']:.4f} ± {model_results['train']['clf_bal_acc_std']:.4f}")
        
        model_results['train']['clf_recall_scores'] = np.array(train_recall_scores)
        model_results['train']['clf_recall_mean'] = np.mean(train_recall_scores)
        model_results['train']['clf_recall_std'] = np.std(train_recall_scores)
        
        model_results['train']['clf_precision_scores'] = np.array(train_precision_scores)
        model_results['train']['clf_precision_mean'] = np.mean(train_precision_scores)
        model_results['train']['clf_precision_std'] = np.std(train_precision_scores)
        
        model_results['train']['clf_mcc_scores'] = np.array(train_mcc_scores)
        model_results['train']['clf_mcc_mean'] = np.mean(train_mcc_scores)
        model_results['train']['clf_mcc_std'] = np.std(train_mcc_scores)

           # Store test classification metrics
        model_results['test']['clf_acc_scores'] = np.array(test_acc_scores)
        model_results['test']['clf_acc_mean'] = np.mean(test_acc_scores)
        model_results['test']['clf_acc_std'] = np.std(test_acc_scores)
        print(f"  Train Accuracy: {model_results['test']['clf_acc_mean']:.4f} ± {model_results['test']['clf_acc_std']:.4f}")

        model_results['test']['clf_f1_scores'] = np.array(test_f1_macro_scores )
        model_results['test']['clf_f1_mean'] = np.mean(test_f1_macro_scores )
        model_results['test']['clf_f1_std'] = np.std(test_f1_macro_scores )
        
        model_results['test']['clf_bal_acc_scores'] = np.array(test_bal_acc_scores)
        model_results['test']['clf_bal_acc_mean'] = np.mean(test_bal_acc_scores)
        model_results['test']['clf_bal_acc_std'] = np.std(test_bal_acc_scores)
        print(f"  Test Balanced Accuracy: {model_results['test']['clf_bal_acc_mean']:.4f} ± {model_results['test']['clf_bal_acc_std']:.4f}")
        
        model_results['test']['clf_recall_scores'] = np.array(test_recall_scores)
        model_results['test']['clf_recall_mean'] = np.mean(test_recall_scores)
        model_results['test']['clf_recall_std'] = np.std(test_recall_scores)
        
        model_results['test']['clf_precision_scores'] = np.array(test_precision_scores)
        model_results['test']['clf_precision_mean'] = np.mean(test_precision_scores)
        model_results['test']['clf_precision_std'] = np.std(test_precision_scores)
        
        model_results['test']['clf_mcc_scores'] = np.array(test_mcc_scores)
        model_results['test']['clf_mcc_mean'] = np.mean(test_mcc_scores)
        model_results['test']['clf_mcc_std'] = np.std(test_mcc_scores)

        
        # Generate overall confusion matrices for test and train data
        test_overall_cm = confusion_matrix(all_test_y_true, all_test_y_pred)
        train_overall_cm = confusion_matrix(all_train_y_true, all_train_y_pred)
        
        confusion_matrices['test'][model_name] = {
                'overall': test_overall_cm,
                'per_fold': test_fold_confusion_matrices
            }
        confusion_matrices['train'][model_name] ={
                'overall': train_overall_cm,
                'per_fold': train_fold_confusion_matrices
            }
        
        
        print(f"\n  Test Overall Confusion Matrix:")
        print(test_overall_cm)
        print(f"\n  Train Overall Confusion Matrix:")
        print(train_overall_cm)
        
      
        cv_results['train'][model_name] = model_results['train']
        cv_results['test'][model_name] = model_results['test']
        fold_predictions[model_name] = fold_preds

    return cv_results, confusion_matrices, fold_predictions,  real_mse_results, penalized_mse_results, balanced_penalized_mse_results, f1_penalized_mse_results, recall_penalized_mse_results, dual_weighted_penalized_mse_results


    # dynamic_penalized_mse_results, dynamic_f1_penalized_mse_results, 

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from sklearn import preprocessing
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    ParameterSampler,
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

os.chdir("/home/linal20/")

df_full = pd.read_csv("GPU_full_sampling4.csv")
df_full=df_full[df_full.batch == 0]


y_type = df_full.iloc[:, 7]
y_time = df_full.iloc[:, 8]
X = df_full.iloc[:, [2,3,4,5,6]]
t0 = df_full.iloc[:, 14]



# from sklearn.model_selection import train_test_split



def preprocess_data(y_type,y_time):

   # Get lengths
   lengths = {
       'y_type': y_type.shape[0],
       'y_time': y_time_train.shape[0]
   }
   
   # Convert categorical to numeric
   def convert_categories(y):
       y = y.copy()  # Create copy to avoid modifying original
       y[y == 'censor'] = 0
       y[y == 'OTB'] = 1
       y[y == 'DBE'] = 2
       return y
   
   # Process all datasets
   processed_data = {
       'y_type': convert_categories(y_type).values,
       'y_time': y_time.values,
       'lengths': lengths
   }
   
   return processed_data

# Usage:
data = preprocess_data(y_type, y_time)

# You can then access the processed data and lengths:
y_type = data['y_type']
y_time = data['y_time']
print(y_type)

def get_optimized_models(X_reg, y_reg, X_clf, y_clf, model_name, n=5, n_iter=500, use_smote=True, random_state=42):

    save_dir = Path("model_search_results")
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp2 = datetime.now().strftime("%Y%m%d")
    search_version = "sparse_v2"
    
    label_encoder = LabelEncoder()
    y_clf_encoded = label_encoder.fit_transform(y_clf)
    y_reg = np.asarray(y_reg)
    y_clf_encoded = np.asarray(y_clf_encoded)

    if len(y_reg) != len(y_clf_encoded):
        raise ValueError("Regression and classification data must contain the same observations.")

    clf_cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=random_state)
    reg_cv = KFold(n_splits=n, shuffle=True, random_state=random_state)

    def calculate_real_mse(actual_times, predicted_times, actual_events, predicted_events):
        actual_times = np.asarray(actual_times)
        predicted_times = np.asarray(predicted_times)
        actual_events = np.asarray(actual_events)
        predicted_events = np.asarray(predicted_events)
        penalty = np.ones(len(actual_times), dtype=float)
        penalty[(actual_events == 0) & (predicted_events == 0)] = 0
        squared_error = penalty * (actual_times - predicted_times) ** 2
        return np.mean(squared_error)

    # PARAMETER SPACES

    dt_reg_params = {
        "model__criterion": ["squared_error", "absolute_error", "poisson"],
        "model__splitter": ["best", "random"],
        "model__max_depth": [None, 3, 7, 12, 20, 30, 50],
        "model__min_samples_split": [2, 5, 10, 20, 40],
        "model__min_samples_leaf": [1, 3, 7, 15, 30],
        "model__max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75, 1.0],
        "model__max_leaf_nodes": [None, 20, 50, 100, 200],
        "model__min_impurity_decrease": [0.0, 1e-5, 1e-3, 1e-2],
    }
    
    dt_clf_params = {
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__splitter": ["best", "random"],
        "model__max_depth": [None, 3, 7, 12, 20, 30, 50],
        "model__min_samples_split": [2, 5, 10, 20, 40],
        "model__min_samples_leaf": [1, 3, 7, 15, 30],
        "model__max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75, 1.0],
        "model__max_leaf_nodes": [None, 20, 50, 100, 200],
        "model__min_impurity_decrease": [0.0, 1e-5, 1e-3, 1e-2],
        "model__class_weight": [None, "balanced"],
    }
    
    rf_reg_params = {
        "model__n_estimators": [100, 300, 600, 1000],
        "model__criterion": ["squared_error", "absolute_error", "poisson"],
        "model__max_depth": [None, 5, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10, 20, 40],
        "model__min_samples_leaf": [1, 3, 7, 15, 30],
        "model__max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75, 1.0],
        "model__bootstrap": [True, False],
    }
    
    rf_clf_params = {
        "model__n_estimators": [100, 300, 600, 1000],
        "model__criterion": ["gini", "entropy", "log_loss"],
        "model__max_depth": [None, 5, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10, 20, 40],
        "model__min_samples_leaf": [1, 3, 7, 15, 30],
        "model__max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75, 1.0],
        "model__bootstrap": [True, False],
        "model__class_weight": [None, "balanced"],
    }
    
    gb_reg_params = {
        "model__n_estimators": [50, 150, 300, 600, 1000],
        "model__learning_rate": [0.005, 0.02, 0.05, 0.1, 0.2, 0.3],
        "model__max_depth": [1, 2, 3, 5, 8, 12],
        "model__min_samples_split": [2, 5, 10, 20, 40],
        "model__min_samples_leaf": [1, 3, 7, 15, 30],
        "model__subsample": [0.5, 0.7, 0.85, 1.0],
        "model__max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75, 1.0],
        "model__loss": ["squared_error", "huber", "absolute_error", "quantile"],
    }
    
    gb_clf_params = {
        "model__n_estimators": [50, 150, 300, 600, 1000],
        "model__learning_rate": [0.005, 0.02, 0.05, 0.1, 0.2, 0.3],
        "model__max_depth": [1, 2, 3, 5, 8, 12],
        "model__min_samples_split": [2, 5, 10, 20, 40],
        "model__min_samples_leaf": [1, 3, 7, 15, 30],
        "model__subsample": [0.5, 0.7, 0.85, 1.0],
        "model__max_features": [None, "sqrt", "log2", 0.25, 0.5, 0.75, 1.0],
        "model__criterion": ["friedman_mse", "squared_error"],
    }
  
    knn_reg_params = {
        "model__n_neighbors": [3, 5, 10, 20, 40, 60, 100],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
        "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "model__leaf_size": [10, 30, 60, 100],
    }
    
    knn_clf_params = {
        "model__n_neighbors": [3, 5, 10, 20, 40, 60, 100],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
        "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "model__leaf_size": [10, 30, 60, 100],
    }

    # SELECT MODEL

    if model_name.lower() == "dt":
        clf_model = DecisionTreeClassifier(random_state=random_state)
        reg_model = DecisionTreeRegressor(random_state=random_state)
        clf_params = dt_clf_params
        reg_params = dt_reg_params

    elif model_name.lower() == "rf":
        clf_model = RandomForestClassifier(random_state=random_state, n_jobs=1)
        reg_model = RandomForestRegressor(random_state=random_state, n_jobs=1)
        clf_params = rf_clf_params
        reg_params = rf_reg_params

    elif model_name.lower() == "gb":
        clf_model = GradientBoostingClassifier(random_state=random_state)
        reg_model = GradientBoostingRegressor(random_state=random_state)
        clf_params = gb_clf_params
        reg_params = gb_reg_params
        
    elif model_name.lower() == "knn":

        clf_model = KNeighborsClassifier()
        reg_model = KNeighborsRegressor()
        clf_params = knn_clf_params
        reg_params = knn_reg_params

    else:
        raise ValueError("model_name must be 'dt', 'rf', or 'gb'")

    # CHECKPOINTED RANDOM SEARCH

    def checkpoint_random_search(pipeline, param_distributions, n_iter, X, y, scoring, cv, checkpoint_file, best_model_file, search_name, random_state=42):

        print("\n" + "=" * 70)
        print(search_name)
        print("=" * 70)

        parameter_list = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=random_state))
        total_parameters = len(parameter_list)

        print(f"Total random combinations: {total_parameters}")

        if checkpoint_file.exists():
            checkpoint = joblib.load(checkpoint_file)
            completed_results = checkpoint.get("completed_results", [])
            best_score = checkpoint.get("best_score", -np.inf)
            best_params = checkpoint.get("best_params", None)

            print("\nCheckpoint found.")
            print(f"Completed: {len(completed_results)} / {total_parameters}")
            print(f"Best score: {best_score:.6f}")
            print(f"Best parameters: {best_params}")

        else:
            completed_results = []
            best_score = -np.inf
            best_params = None

        completed_keys = set()

        for result in completed_results:
            param_key = tuple(sorted(result["params"].items()))
            completed_keys.add(param_key)

        for i, params in enumerate(parameter_list, start=1):

            param_key = tuple(sorted(params.items()))

            if param_key in completed_keys:
                print(f"[{i}/{total_parameters}] Already completed.")
                continue

            print("\n" + "-" * 70)
            print(f"[{i}/{total_parameters}] Random Search")
            print(f"Parameters:\n{params}")

            current_model = clone(pipeline)
            current_model.set_params(**params)

            scores = cross_val_score(current_model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
            mean_score = np.mean(scores)

            print(f"Fold scores: {scores}")
            print(f"Mean score: {mean_score:.6f}")

            result = {"iteration": i, "params": params, "score": mean_score, "fold_scores": scores}
            completed_results.append(result)
            completed_keys.add(param_key)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

                print("\n*** NEW BEST MODEL ***")
                print(f"Best score: {best_score:.6f}")
                print(f"Best params:\n{best_params}")

                best_model = clone(pipeline)
                best_model.set_params(**best_params)
                best_model.fit(X, y)
                joblib.dump(best_model, best_model_file)

            checkpoint = {
                "search_type": "random_search",
                "model_name": model_name,
                "completed_results": completed_results,
                "completed_count": len(completed_results),
                "total_count": total_parameters,
                "best_score": best_score,
                "best_params": best_params,
                "n_iter": n_iter,
                "random_state": random_state
            }

            joblib.dump(checkpoint, checkpoint_file)
            print("Checkpoint saved.")

        if best_model_file.exists():
            best_model = joblib.load(best_model_file)
        else:
            if best_params is None:
                raise RuntimeError("No successful model was found.")

            best_model = clone(pipeline)
            best_model.set_params(**best_params)
            best_model.fit(X, y)
            joblib.dump(best_model, best_model_file)

        return {
            "best_model": best_model,
            "best_params": best_params,
            "best_score": best_score,
            "completed_results": completed_results
        }

    # =====================================================
    # CLASSIFICATION PIPELINE
    # =====================================================

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} CLASSIFICATION RANDOM SEARCH")
    print("=" * 70)

    if use_smote:
        clf_pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("model", clf_model)
        ])
    else:
        clf_pipeline = Pipeline([
            ("model", clf_model)
        ])

    clf_checkpoint_file = save_dir / f"{model_name}_classifier_random_checkpoint_{search_version}_{timestamp2}.joblib"
    clf_best_model_file = save_dir / f"{model_name}_classifier_random_best_model_{search_version}_{timestamp2}.joblib"

    clf_search = checkpoint_random_search(
        pipeline=clf_pipeline,
        param_distributions=clf_params,
        n_iter=n_iter,
        X=X_clf,
        y=y_clf_encoded,
        scoring="balanced_accuracy",
        cv=clf_cv,
        checkpoint_file=clf_checkpoint_file,
        best_model_file=clf_best_model_file,
        search_name=f"{model_name.upper()} Classification",
        random_state=random_state
    )

    best_clf_model = clf_search["best_model"]
    best_clf_params = clf_search["best_params"]
    best_clf_score = clf_search["best_score"]

    # =====================================================
    # CLASSIFICATION OOF PREDICTIONS
    # =====================================================

    print("\nGenerating classification OOF predictions...")

    predicted_events = cross_val_predict(
        clone(best_clf_model), X_clf, y_clf_encoded,
        cv=clf_cv, method="predict", n_jobs=-1
    )

    # =====================================================
    # REGRESSION PIPELINE
    # =====================================================

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} REGRESSION RANDOM SEARCH")
    print("=" * 70)

    reg_pipeline = Pipeline([
        ("model", reg_model)
    ])

    reg_checkpoint_file = save_dir / f"{model_name}_regressor_random_checkpoint_{timestamp2}.joblib"
    reg_best_model_file = save_dir / f"{model_name}_regressor_random_best_model_{timestamp2}.joblib"

    reg_search = checkpoint_random_search(
        pipeline=reg_pipeline,
        param_distributions=reg_params,
        n_iter=n_iter,
        X=X_reg,
        y=y_reg,
        scoring="neg_mean_squared_error",
        cv=reg_cv,
        checkpoint_file=reg_checkpoint_file,
        best_model_file=reg_best_model_file,
        search_name=f"{model_name.upper()} Regression",
        random_state=random_state
    )

    best_reg_model = reg_search["best_model"]
    best_reg_params = reg_search["best_params"]
    best_reg_cv_mse = -reg_search["best_score"]

    # =====================================================
    # REGRESSION OOF PREDICTIONS
    # =====================================================

    print("\nGenerating regression OOF predictions...")

    predicted_times = cross_val_predict(
        clone(best_reg_model), X_reg, y_reg,
        cv=reg_cv, method="predict", n_jobs=-1
    )


    # REAL MSE


    real_mse = calculate_real_mse(
        actual_times=y_reg,
        predicted_times=predicted_times,
        actual_events=y_clf_encoded,
        predicted_events=predicted_events
    )


    # FINAL RESULTS

    print("\n" + "=" * 70)
    print(f"FINAL {model_name.upper()} RESULTS")
    print("=" * 70)

    print("\nBest classification parameters:")
    print(best_clf_params)
    print(f"Balanced Accuracy: {best_clf_score:.6f}")

    print("\nBest regression parameters:")
    print(best_reg_params)
    print(f"Regression CV MSE: {best_reg_cv_mse:.6f}")
    print(f"Real MSE: {real_mse:.6f}")

    # FINAL SUMMARY

    final_results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "search_type": "random_search",
        "n_iter": n_iter,
        "cv_folds": n,
        "random_state": random_state,
        "use_smote": use_smote,
        "best_clf_params": best_clf_params,
        "best_clf_score": best_clf_score,
        "best_reg_params": best_reg_params,
        "best_reg_cv_mse": best_reg_cv_mse,
        "real_mse": real_mse,
        "classifier_checkpoint": str(clf_checkpoint_file),
        "regressor_checkpoint": str(reg_checkpoint_file),
        "classifier_best_model": str(clf_best_model_file),
        "regressor_best_model": str(reg_best_model_file)
    }

    final_results_file = save_dir / f"{model_name}_random_final_results_{timestamp}.joblib"
    joblib.dump(final_results, final_results_file)

    print("\nFinal results saved to:")
    print(final_results_file)


    return {
        "best_clf_name": model_name,
        "best_clf_model": best_clf_model,
        "best_clf_params": best_clf_params,
        "best_clf_score": best_clf_score,
        "best_reg_name": model_name,
        "best_reg_model": best_reg_model,
        "best_reg_params": best_reg_params,
        "best_reg_cv_mse": best_reg_cv_mse,
        "best_reg_score": real_mse,
        "predicted_events": predicted_events,
        "predicted_times": predicted_times,
        "label_encoder": label_encoder,
        "clf_search_results": clf_search["completed_results"],
        "reg_search_results": reg_search["completed_results"]
    }

dt_results = get_optimized_models(
    X_reg=X, y_reg=y_time,
    X_clf=X, y_clf=y_type,
    model_name="dt", n=5, n_iter=200,
    use_smote=True, random_state=42
)

rf_results = get_optimized_models(
    X_reg=X, y_reg=y_time,
    X_clf=X, y_clf=y_type,
    model_name="rf", n=5, n_iter=200,
    use_smote=True, random_state=42
)


KNN_results = get_optimized_models(
    X_reg=X_full_scaled, y_reg=y_time,
    X_clf=X_full_scaled, y_clf=y_type,
    model_name="knn", n=5, n_iter=200,
    use_smote=True, random_state=42
)


gb_results = get_optimized_models(
    X_reg=X_full_scaled, y_reg=y_time,
    X_clf=X_full_scaled, y_clf=y_type,
    model_name="gb", n=5, n_iter=200,
    use_smote=True, random_state=42
)
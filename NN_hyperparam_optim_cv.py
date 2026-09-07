import os, copy, random, joblib, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from itertools import product
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score




def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("=" * 75)
    print("GPU ENABLED:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda, "| PyTorch:", torch.__version__)
    print("=" * 75)
else:
    device = torch.device("cpu")
    print("=" * 75)
    print("WARNING: CUDA NOT AVAILABLE — USING CPU")
    print("=" * 75)

# 2. LOAD DATA

os.chdir("/home/linal20/")

df_full = pd.read_csv("GPU_dat.csv")
df_full = df_full[df_full.batch == 0].copy()

y_type = df_full.iloc[:, 7].copy()
y_time = df_full.iloc[:, 8].copy()
X = df_full.iloc[:, [2, 3, 4, 5, 6, 16]].copy()
t0 = df_full.iloc[:, 14].copy()



# 3. TRAIN / TEST SPLIT


X_train, X_test, y_type_train, y_type_test, y_time_train, y_time_test, t0_train, t0_test = train_test_split(
    X, y_type, y_time, t0, random_state=114, test_size=0.25, shuffle=False
)

print("Training observations:", len(X_train))
print("Testing observations :", len(X_test))



# 4. FEATURES

categorical_cols = ["col", "row", "cage", "slot", "node"]
numeric_cols = ["t2"]


# 5. EMBEDDING PREPROCESSING


def prepare_embedding_data(X_train, X_test):
    Xtr, Xte, mappings = X_train.copy(), X_test.copy(), {}

    for col in categorical_cols:
        categories = pd.Index(Xtr[col].dropna().unique())
        # print(categories)
        mapping = {value: i for i, value in enumerate(categories)}
        # print(mapping)
        mappings[col] = mapping

        # print(Xtr[col])
        Xtr[col] = Xtr[col].map(mapping)
        Xte[col] = Xte[col].map(mapping)
        # print(Xtr[col])
        if Xtr[col].isna().any(): raise ValueError(f"Missing category in training data: {col}")
        if Xte[col].isna().any(): raise ValueError(f"Unseen category in test data: {col}")

        Xtr[col] = Xtr[col].astype(np.int64)
        Xte[col] = Xte[col].astype(np.int64)

    scaler = StandardScaler()
    Xtr[numeric_cols] = scaler.fit_transform(Xtr[numeric_cols])
    Xte[numeric_cols] = scaler.transform(Xte[numeric_cols])

    return Xtr, Xte, mappings, scaler


X_train_emb, X_test_emb, category_maps, t2_scaler = prepare_embedding_data(X_train, X_test)

cat_dims = [len(category_maps[col]) for col in categorical_cols]
num_numeric_features = len(numeric_cols)

print("Categorical dimensions:", cat_dims)
print("Numeric features:", num_numeric_features)
# 6. OUTCOME ENCODING

def encode_type(y):
    return y.map({"censor": 0, "OTB": 1, "DBE": 2}).astype(np.int64).values


y_type_train_np = encode_type(y_type_train)
y_type_test_np = encode_type(y_type_test)

y_time_train_np = y_time_train.astype(np.float32).values
y_time_test_np = y_time_test.astype(np.float32).values


# 7. ACTIVATIONS

ACTIVATION_MAP = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


# 8. TIME MODEL

class CategoricalEmbeddingNN_time(nn.Module):

    def __init__(
        self, cat_dims, num_numeric_features=0,
        embedding_dims=(5, 4, 2, 4, 3),
        hidden_units=(128, 64), dropout_rate=0.2,
        activations=("relu", "relu")
    ):
        super().__init__()

        if len(embedding_dims) != len(cat_dims):
            raise ValueError(
                "Number of embedding dimensions must equal number of categorical variables."
            )

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat, emb_dim)
            for num_cat, emb_dim in zip(cat_dims, embedding_dims)
        ])

        total_embed_size = sum(emb.embedding_dim for emb in self.embeddings)
        input_dim = total_embed_size + num_numeric_features

        if len(activations) != len(hidden_units):
            raise ValueError(
                "Number of activations must equal number of hidden layers."
            )

        layers, prev_dim = [], input_dim

        for hidden_dim, activation in zip(hidden_units, activations):
            if activation not in ACTIVATION_MAP:
                raise ValueError(f"Unknown activation: {activation}")

            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                ACTIVATION_MAP[activation](),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.layers = nn.Sequential(*layers)

        for emb in self.embeddings:
            nn.init.kaiming_normal_(emb.weight)

    def forward(self, x_numeric, x_categorical):
        embedded = torch.cat(
            [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)],
            dim=1
        )
        return self.layers(torch.cat([x_numeric, embedded], dim=1))


# 9. TYPE MODEL

class CategoricalEmbeddingNN_type(nn.Module):

    def __init__(
        self, cat_dims, num_numeric_features=0,
        embedding_dims=(5, 4, 2, 4, 3),
        hidden_units=(128, 64), dropout_rate=0.2,
        activations=("relu", "relu")
    ):
        super().__init__()

        if len(embedding_dims) != len(cat_dims):
            raise ValueError(
                "Number of embedding dimensions must equal number of categorical variables."
            )

        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim)
            for cat_dim, emb_dim in zip(cat_dims, embedding_dims)
        ])

        total_embed_size = sum(emb.embedding_dim for emb in self.embeddings)
        input_dim = total_embed_size + num_numeric_features

        if len(activations) != len(hidden_units):
            raise ValueError(
                "Number of activations must equal number of hidden layers."
            )

        layers, prev_dim = [], input_dim

        for hidden_dim, activation in zip(hidden_units, activations):
            if activation not in ACTIVATION_MAP:
                raise ValueError(f"Unknown activation: {activation}")

            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                ACTIVATION_MAP[activation](),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.type_layers = nn.Sequential(*layers)
        self.type_output = nn.Linear(prev_dim, 3)

    def forward(self, x_numeric, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_numeric], dim=1)
        return self.type_output(self.type_layers(x))


# 10. EMBEDDING CONFIGURATIONS + ARCHITECTURES

embedding_configs = [
    [3, 3, 2, 3, 2],      # Small
    [5, 4, 2, 4, 3],      # Medium
    [8, 6, 3, 6, 4],      # Large
    [12, 8, 3, 8, 4],     # Very large
    [16, 10, 3, 10, 5],   # Extra large
]

embedding_config_names = [
    "small", "medium", "large", "very_large", "Extra_large"
]

embedding_config_lookup = dict(zip(embedding_config_names, embedding_configs))

hidden_architectures = [
    (64,), (128,), (64, 32), (128, 64),
    (256, 128), (128, 64, 32), (256, 128, 64)
]

print("\n" + "=" * 75)
print("EMBEDDING CONFIGURATIONS")
print("=" * 75)

for name, dims in zip(embedding_config_names, embedding_configs):
    print(f"{name:12s}: {dims} | total={sum(dims)}")


# 11. ACTIVATION COMBINATIONS

activation_choices = ["relu", "gelu", "silu"]

activation_combinations = {
    architecture: list(product(activation_choices, repeat=len(architecture)))
    for architecture in hidden_architectures
}

for architecture in hidden_architectures:
    print(
        f"{architecture}: "
        f"{len(activation_combinations[architecture])} activation combinations"
    )


# 12. RANDOM SEARCH SPACE

random_param_space = {
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lr": [1e-4, 3e-4, 5e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    "batch_size": [64, 128, 256, 512],
    "max_epochs": [50, 100, 150, 200]
}

n_random_iter = 100
random_configs = {}

for arch_idx, architecture in enumerate(hidden_architectures):
    space = {
        **random_param_space,
        "activations": activation_combinations[architecture]
    }

    random_configs[architecture] = list(
        ParameterSampler(space, n_iter=n_random_iter, random_state=42 + arch_idx)
    )

print("\nRandom configurations per architecture:", n_random_iter)


# 13. BASELINE CLASS WEIGHTS

def get_baseline_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return classes, counts, weights.astype(np.float32)


baseline_classes, baseline_counts, baseline_weights = \
    get_baseline_class_weights(y_type_train_np)

print("\n" + "=" * 75)
print("BASELINE CLASS WEIGHTS")
print("=" * 75)

print("Classes:", baseline_classes)
print("Counts:", baseline_counts)
print("Weights:", baseline_weights)
print("Mean:", baseline_weights.mean())



# 15. TENSOR FUNCTION

def make_tensors(X_df, y=None):
    x_num = torch.tensor(X_df[numeric_cols].values, dtype=torch.float32)
    x_cat = torch.tensor(X_df[categorical_cols].values, dtype=torch.long)

    if y is None:
        return x_num, x_cat

    return x_num, x_cat, torch.tensor(np.asarray(y))


# 16. FOLD BASELINE WEIGHTS

def get_fold_class_weights(y_train, multipliers):
    classes, counts = np.unique(y_train, return_counts=True)
    n_samples, n_classes = len(y_train), len(classes)
    baseline = n_samples / (n_classes * counts)

    final_weights = baseline * np.asarray(multipliers, dtype=np.float32)
    return final_weights.astype(np.float32)


# 17. TIME CV

def evaluate_time_config(
    X, y, cv, embedding_dims, hidden_units, config, patience=15
):
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
        print(f"        Fold {fold}/{cv.n_splits}", end=" | ")
        set_seed(42 + fold)

        Xtr, Xva = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yva = y[train_idx], y[val_idx]

        xnum_tr, xcat_tr, ytr_t = make_tensors(Xtr, ytr)
        xnum_va, xcat_va, yva_t = make_tensors(Xva, yva)

        xnum_tr, xcat_tr = xnum_tr.to(device), xcat_tr.to(device)
        xnum_va, xcat_va = xnum_va.to(device), xcat_va.to(device)

        ytr_t = ytr_t.float().reshape(-1, 1).to(device)
        yva_t = yva_t.float().reshape(-1, 1).to(device)

        model = CategoricalEmbeddingNN_time(
            cat_dims=cat_dims,
            num_numeric_features=num_numeric_features,
            embedding_dims=embedding_dims,
            hidden_units=hidden_units,
            dropout_rate=config["dropout_rate"],
            activations=config["activations"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        criterion = nn.MSELoss()
        best_loss, best_state, no_improve = np.inf, None, 0

        batch_size = config["batch_size"]
        max_epochs = config["max_epochs"]
        n = len(xnum_tr)

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n, device=device)

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]

                optimizer.zero_grad()
                pred = model(xnum_tr[idx], xcat_tr[idx])
                loss = criterion(pred, ytr_t[idx])
                loss.backward()
                optimizer.step()

            model.eval()

            with torch.no_grad():
                val_pred = model(xnum_va, xcat_va)
                val_loss = criterion(val_pred, yva_t).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            pred = model(xnum_va, xcat_va).cpu().numpy().ravel()

        rmse = np.sqrt(mean_squared_error(yva, pred))
        mae = mean_absolute_error(yva, pred)
        r2 = r2_score(yva, pred)

        fold_results.append({
            "fold": fold,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "epochs_used": epoch + 1
        })

        print(f"RMSE={rmse:.5f}")

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "mean_rmse": np.mean([r["rmse"] for r in fold_results]),
        "mean_mae": np.mean([r["mae"] for r in fold_results]),
        "mean_r2": np.mean([r["r2"] for r in fold_results]),
        "mean_epochs": np.mean([r["epochs_used"] for r in fold_results]),
        "fold_results": fold_results
    }


# 18. TYPE CV

def evaluate_type_config(
    X, y, cv, embedding_dims, hidden_units, config,
    weight_multipliers, patience=15
):
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"        Fold {fold}/{cv.n_splits}", end=" | ")
        set_seed(42 + fold)

        Xtr, Xva = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yva = y[train_idx], y[val_idx]

        xnum_tr, xcat_tr, ytr_t = make_tensors(Xtr, ytr)
        xnum_va, xcat_va, yva_t = make_tensors(Xva, yva)

        xnum_tr, xcat_tr = xnum_tr.to(device), xcat_tr.to(device)
        xnum_va, xcat_va = xnum_va.to(device), xcat_va.to(device)

        ytr_t = ytr_t.long().to(device)
        yva_t = yva_t.long().to(device)

        fold_baseline = get_fold_class_weights(ytr, [1.0, 1.0, 1.0])

        final_class_weights = (
            fold_baseline * np.asarray(weight_multipliers, dtype=np.float32)
        )

        class_weights_t = torch.tensor(
            final_class_weights, dtype=torch.float32, device=device
        )

        model = CategoricalEmbeddingNN_type(
            cat_dims=cat_dims,
            num_numeric_features=num_numeric_features,
            embedding_dims=embedding_dims,
            hidden_units=hidden_units,
            dropout_rate=config["dropout_rate"],
            activations=config["activations"]
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )

        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        best_loss, best_state, no_improve = np.inf, None, 0

        batch_size = config["batch_size"]
        max_epochs = config["max_epochs"]
        n = len(xnum_tr)

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(n, device=device)

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]

                optimizer.zero_grad()
                logits = model(xnum_tr[idx], xcat_tr[idx])
                loss = criterion(logits, ytr_t[idx])
                loss.backward()
                optimizer.step()

            model.eval()

            with torch.no_grad():
                val_logits = model(xnum_va, xcat_va)
                val_loss = criterion(val_logits, yva_t).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            logits = model(xnum_va, xcat_va)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

        bal_acc = balanced_accuracy_score(yva, pred)
        acc = accuracy_score(yva, pred)
        macro_f1 = f1_score(yva, pred, average="macro")

        fold_results.append({
            "fold": fold,
            "balanced_accuracy": bal_acc,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "epochs_used": epoch + 1
        })

        print(f"BA={bal_acc:.5f}")

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "mean_balanced_accuracy": np.mean([r["balanced_accuracy"] for r in fold_results]),
        "mean_accuracy": np.mean([r["accuracy"] for r in fold_results]),
        "mean_macro_f1": np.mean([r["macro_f1"] for r in fold_results]),
        "mean_epochs": np.mean([r["epochs_used"] for r in fold_results]),
        "fold_results": fold_results}


# 19. CHECKPOINT

def save_checkpoint(results, folder, filename):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(results, folder / f"{filename}.pkl")

    flat = [
        {k: v for k, v in r.items() if k != "fold_results"}
        for r in results
    ]

    pd.DataFrame(flat).to_csv(folder / f"{filename}.csv", index=False)


# 20. TIME RANDOM SEARCH

def tune_time(X, y, n_splits=5, patience=15, save_dir="NN_tuning/time"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = save_dir / "time_checkpoint.pkl"

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    total = len(embedding_configs) * len(hidden_architectures) * n_random_iter

    if checkpoint.exists():
        results = joblib.load(checkpoint)

        completed = {
            (tuple(r["embedding_dims"]), r["hidden_units"], r["random_index"])
            for r in results
        }

        print(f"Resuming: {len(results)}/{total}")
    else:
        results, completed = [], set()

    counter = 0

    for arch_idx, hidden in enumerate(hidden_architectures):
        for emb_idx, embedding_dims in enumerate(embedding_configs):
            embedding_name = embedding_config_names[emb_idx]

            for random_idx, config in enumerate(random_configs[hidden]):
                counter += 1
                key = (tuple(embedding_dims), str(hidden), random_idx)

                if key in completed:
                    continue

                print("\n" + "=" * 75)
                print(
                    f"TIME {counter}/{total} | "
                    f"embedding={embedding_name} {embedding_dims} | "
                    f"hidden={hidden}"
                )
                print(
                    f"activations={config['activations']} | "
                    f"dropout={config['dropout_rate']} | "
                    f"lr={config['lr']} | wd={config['weight_decay']} | "
                    f"batch={config['batch_size']} | "
                    f"max_epochs={config['max_epochs']}"
                )

                result = evaluate_time_config(
                    X, y, cv, embedding_dims, hidden, config, patience
                )

                row = {
                    "configuration": counter,
                    "architecture_index": arch_idx,
                    "embedding_config_index": emb_idx,
                    "embedding_config_name": embedding_name,
                    "embedding_dims": list(embedding_dims),
                    "random_index": random_idx,
                    "hidden_units": str(hidden),
                    "activations": str(config["activations"]),
                    **{k: v for k, v in config.items() if k != "activations"},
                    "mean_rmse": result["mean_rmse"],
                    "mean_mae": result["mean_mae"],
                    "mean_r2": result["mean_r2"],
                    "mean_epochs": result["mean_epochs"],
                    "fold_results": result["fold_results"]
                }

                results.append(row)

                save_checkpoint(results, save_dir, "time_checkpoint")

                print(
                    f"MEAN RMSE={result['mean_rmse']:.5f} | "
                    f"MAE={result['mean_mae']:.5f} | "
                    f"R2={result['mean_r2']:.5f}"
                )

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "fold_results"}
        for r in results
    ])

    results_df = (
        results_df
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )

    results_df.to_csv(save_dir / "time_tuning_ranked.csv", index=False)
    joblib.dump(results, save_dir / "time_tuning_results.pkl")

    print("\n" + "=" * 75)
    print("BEST TIME MODEL")
    print("=" * 75)
    print(results_df.iloc[0])

    return results_df, results



# 21. Tune TYPE 

def tune_type(X, y, n_splits=5, patience=15, save_dir="NN_tuning/type"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = save_dir / "type_checkpoint.pkl"

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    total = len(embedding_configs) * len(hidden_architectures) * n_random_iter
    baseline_multipliers = [1.0, 1.0, 1.0]

    if checkpoint.exists():
        results = joblib.load(checkpoint)
        completed = {
            (tuple(r["embedding_dims"]), r["hidden_units"], r["random_index"])
            for r in results
        }
        print(f"Resuming: {len(results)}/{total}")
    else:
        results, completed = [], set()

    counter = 0

    for arch_idx, hidden in enumerate(hidden_architectures):
        for emb_idx, embedding_dims in enumerate(embedding_configs):
            embedding_name = embedding_config_names[emb_idx]

            for random_idx, config in enumerate(random_configs[hidden]):
                counter += 1
                key = (tuple(embedding_dims), str(hidden), random_idx)

                if key in completed:
                    continue

                print("\n" + "=" * 75)
                print(
                    f"TYPE {counter}/{total} | "
                    f"embedding={embedding_name} {embedding_dims} | hidden={hidden}"
                )
                print(
                    f"activations={config['activations']} | "
                    f"dropout={config['dropout_rate']} | lr={config['lr']} | "
                    f"wd={config['weight_decay']} | batch={config['batch_size']} | "
                    f"max_epochs={config['max_epochs']}"
                )
                print("class weights = inverse-frequency baseline")

                result = evaluate_type_config(
                    X, y, cv, embedding_dims, hidden, config,
                    baseline_multipliers, patience
                )

                row = {
                    "configuration": counter,
                    "architecture_index": arch_idx,
                    "embedding_config_index": emb_idx,
                    "embedding_config_name": embedding_name,
                    "embedding_dims": list(embedding_dims),
                    "random_index": random_idx,
                    "hidden_units": str(hidden),
                    "activations": str(config["activations"]),
                    **{k: v for k, v in config.items() if k != "activations"},
                    "weight_multiplier_w0": 1.0,
                    "weight_multiplier_w1": 1.0,
                    "weight_multiplier_w2": 1.0,
                    "mean_balanced_accuracy": result["mean_balanced_accuracy"],
                    "mean_accuracy": result["mean_accuracy"],
                    "mean_macro_f1": result["mean_macro_f1"],
                    "mean_epochs": result["mean_epochs"],
                    "fold_results": result["fold_results"],
                }

                results.append(row)

                save_checkpoint(
                    results, save_dir, "type_checkpoint"
                )

                print(
                    f"MEAN BA={result['mean_balanced_accuracy']:.5f} | "
                    f"F1={result['mean_macro_f1']:.5f}"
                )

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "fold_results"}
        for r in results
    ])

    results_df = (
        results_df
        .sort_values("mean_balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )

    results_df.to_csv(save_dir / "type_ranked.csv", index=False)
    joblib.dump(results, save_dir / "type_results.pkl")

    print("\n" + "=" * 75)
    print("BEST TIME MODEL")
    print("=" * 75)
    
    print(results_df.iloc[0])

    return results_df, results


time_df, time_results = tune_time(
    X_train_emb,
    y_time_train_np,
    n_splits=3,
    patience=15,
    save_dir="NN_tuning/time"
)

type_df, type_results = tune_type(
    X_train_emb,
    y_type_train_np,
    n_splits=3,
    patience=15,
    save_dir="NN_tuning/type"
)


import numpy as np
import pandas as pd
import itertools
from pathlib import Path


# Checkpoint settings

CHECKPOINT_DIR = Path("weight_search_checkpoint_[0.3, 0.4, 0.5]")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = CHECKPOINT_DIR / "weight_search_results_balanced_penalized_mse.csv"


# Candidate additive adjustments

weight1_values = [0.0, 0.1]
weight2_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
weight3_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


# Load existing checkpoint

if CHECKPOINT_FILE.exists():
    weight_results = pd.read_csv(CHECKPOINT_FILE)
    print(f"Loaded checkpoint: {len(weight_results)} combinations already completed.")
else:
    weight_results = pd.DataFrame(columns=[
        "weight_1", "weight_2", "weight_3", "balanced_penalized_mse"
    ])
    print("No checkpoint found. Starting from scratch.")


# Completed combinations

completed_weights = set(zip(
    weight_results["weight_1"],
    weight_results["weight_2"],
    weight_results["weight_3"]
))


# Total combinations

all_combinations = list(itertools.product(
    weight1_values, weight2_values, weight3_values
))

total_combinations = len(all_combinations)

print(f"Total combinations: {total_combinations}")
print(f"Remaining combinations: {total_combinations - len(completed_weights)}")

#####################
# Run weight search #
#####################
for i, (w1, w2, w3) in enumerate(all_combinations, start=1):
    weights = [w1, w2, w3]

    if (w1, w2, w3) in completed_weights:
        print(f"[{i}/{total_combinations}] SKIP {weights} — already completed")
        continue

    print(f"\n[{i}/{total_combinations}] Running weights = {weights}")

    try:
        results = run_best_configuration_cv(weights=weights, n_folds=3)

        score = results["combined"]["summary"]["balanced_penalized_mse"]["mean"]

        new_result = pd.DataFrame([{
            "weight_1": w1,
            "weight_2": w2,
            "weight_3": w3,
            "balanced_penalized_mse": score
        }])

        weight_results = pd.concat(
            [weight_results, new_result],
            ignore_index=True
        )

        # Save checkpoint immediately
        weight_results.to_csv(CHECKPOINT_FILE, index=False)
        completed_weights.add((w1, w2, w3))

        print(
            f"Completed: weights={weights} | "
            f"Balanced Penalized MSE={score:.6f}"
        )
        print(
            f"Checkpoint saved: "
            f"{len(weight_results)}/{total_combinations}"
        )

    except Exception as e:
        print(f"ERROR for weights={weights}: {e}")
        continue


# Sort results: lower Balanced Penalized MSE = better

weight_results = (
    weight_results
    .sort_values("balanced_penalized_mse", ascending=True)
    .reset_index(drop=True)
)


# Save final sorted results

weight_results.to_csv(CHECKPOINT_FILE, index=False)


# Display top results

print("\nTop weight combinations:")
print(weight_results.head(10).to_string(index=False))


# Best weights

if len(weight_results) > 0:
    best_weights = weight_results.loc[
        0, ["weight_1", "weight_2", "weight_3"]
    ].tolist()

    best_score = weight_results.loc[0, "balanced_penalized_mse"]

    print("\nBEST WEIGHTS:")
    print(best_weights)
    print(f"Best Balanced Penalized MSE: {best_score:.6f}")
# Code Citations

## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
```


## License: MIT
https://github.com/mokronos/nonlinearLSTM/blob/444447db3d9ba8de741f75a265c9a42d8e04740a/main.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train
```


## License: unknown
https://github.com/DataScienceHamburg/PyTorchUltimateMaterial/blob/8ee30716a1c7406f5b6b6d924a0c5319224bdafc/180_LSTM/FunctionApproximation_incl_extrapolation_end.py

```


Based on your loaded dataset, here's the full EDA and preprocessing pipeline. Add these cells one by one after your data loading cell:

**Cell 2 — Basic Info**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst few rows:")
df.head(10)
````

**Cell 3 — Detailed Column Summary**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
print("Columns and dtypes:")
print("-" * 80)
for col in df.columns:
    print(f"  {col:30s} | dtype: {str(df[col].dtype):10s} | unique: {df[col].nunique():6d} | nulls: {df[col].isna().sum():6d} ({df[col].isna().mean()*100:.2f}%)")
````

**Cell 4 — Descriptive Statistics**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
df.describe()
````

**Cell 5 — Identify Column Types**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import numpy as np

# Detect date columns
date_cols = []
for col in df.columns:
    if df[col].dtype == 'object':
        try:
            pd.to_datetime(df[col].dropna().head(100))
            date_cols.append(col)
        except:
            pass

categorical_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col not in date_cols]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Date columns:        {date_cols}")
print(f"Categorical columns: {categorical_cols}")
print(f"Numeric columns:     {numeric_cols}")
print(f"\nTotal: {len(date_cols)} date + {len(categorical_cols)} categorical + {len(numeric_cols)} numeric = {len(date_cols)+len(categorical_cols)+len(numeric_cols)}")
````

**Cell 6 — Parse Date Column**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
if len(date_cols) > 0:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    print(f"Parsed '{date_col}' as datetime.")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Duration: {df[date_col].max() - df[date_col].min()}")
else:
    date_col = None
    print("No date column detected.")

df.head()
````

**Cell 7 — Missing Values Visualization**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Bar chart of missing values
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    missing.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Missing Values Count per Column')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
else:
    axes[0].text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=16)
    axes[0].set_title('Missing Values Count per Column')

# Missing values heatmap
sample_df = df.sample(min(1000, len(df)), random_state=42) if len(df) > 1000 else df
sns.heatmap(sample_df.isna(), cbar=True, yticklabels=False, ax=axes[1], cmap='viridis')
axes[1].set_title('Missing Values Heatmap (sampled)')

plt.tight_layout()
plt.show()
````

**Cell 8 — Categorical Column Analysis**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
for col in categorical_cols:
    print(f"\n{'='*50}")
    print(f"Column: {col}  |  Unique: {df[col].nunique()}")
    print(f"{'='*50}")
    vc = df[col].value_counts()
    print(vc.head(20))
    
    if df[col].nunique() <= 30:
        fig, ax = plt.subplots(figsize=(12, 4))
        vc.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(f'Value Counts: {col}')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
````

**Cell 9 — Numeric Distributions**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
n_numeric = len(numeric_cols)
n_cols_plot = 3
n_rows_plot = (n_numeric + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    df[col].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title(f'{col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution of Numeric Features', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 10 — Box Plots for Outliers**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6 * n_cols_plot, 4 * n_rows_plot))
axes = axes.flatten() if n_numeric > 1 else [axes]

for i, col in enumerate(numeric_cols):
    ax = axes[i]
    sns.boxplot(x=df[col].dropna(), ax=ax, color='lightcoral')
    ax.set_title(f'{col}')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Box Plots - Outlier Detection', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
````

**Cell 11 — Correlation Heatmap**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={'size': 8})
ax.set_title('Correlation Heatmap (Numeric Features)', fontsize=16)
plt.tight_layout()
plt.show()
````

**Cell 12 — Pairplot**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
pair_cols = numeric_cols[:6]
sample_pair = df[pair_cols].dropna().sample(min(2000, len(df)), random_state=42)

g = sns.pairplot(sample_pair, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10})
g.figure.suptitle('Pairplot of Key Numeric Features', y=1.02, fontsize=16)
plt.show()
````

**Cell 13 — Target & Feature Selection + Cleaning**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
# Identify AQI target column
target_candidates = [c for c in numeric_cols if 'aqi' in c.lower()]
TARGET_COL = target_candidates[0] if target_candidates else numeric_cols[-1]
FEATURE_COLS = [c for c in numeric_cols if c != TARGET_COL]

print(f"Target column:  {TARGET_COL}")
print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nTarget stats:\n{df[TARGET_COL].describe()}")

# Grouping column for station/city-level imputation
group_col_for_impute = None
for c in categorical_cols:
    if any(k in c.lower() for k in ['station', 'city', 'site']):
        group_col_for_impute = c
        break

print(f"\nGroup column for imputation: {group_col_for_impute}")

# Build clean dataframe
all_cols_needed = FEATURE_COLS + [TARGET_COL]
if date_col:
    all_cols_needed = [date_col] + all_cols_needed
if group_col_for_impute:
    all_cols_needed = [group_col_for_impute] + all_cols_needed
all_cols_needed = list(dict.fromkeys(all_cols_needed))

df_clean = df[all_cols_needed].copy()
print(f"\nBefore cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum()}\n")

# Drop rows where target is NaN
df_clean = df_clean.dropna(subset=[TARGET_COL])
print(f"After dropping NaN target: {df_clean.shape}")

# Forward fill + backward fill within groups
if group_col_for_impute:
    df_clean[FEATURE_COLS] = df_clean.groupby(group_col_for_impute)[FEATURE_COLS].transform(
        lambda x: x.ffill().bfill()
    )
else:
    df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].ffill().bfill()

# Fill remaining NaN with median
df_clean[FEATURE_COLS] = df_clean[FEATURE_COLS].fillna(df_clean[FEATURE_COLS].median())

# Drop any remaining NaN
df_clean = df_clean.dropna(subset=FEATURE_COLS + [TARGET_COL])

print(f"After imputation & cleanup: {df_clean.shape}")
print(f"Remaining NaN: {df_clean[FEATURE_COLS + [TARGET_COL]].isna().sum().sum()}")
df_clean.head()
````

**Cell 14 — Sequence Creation & DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ===== CONFIG =====
SEQUENCE_LENGTH = 30
FORECAST_HORIZON = 1
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 64

# ===== NORMALIZER =====
class Normalizer:
    def __init__(self, feature_range=(0, 1)):
        self.feature_scaler = MinMaxScaler(feature_range=feature_range)
        self.target_scaler = MinMaxScaler(feature_range=feature_range)
        self._is_fitted = False
    
    def fit(self, features, target):
        self.feature_scaler.fit(features)
        self.target_scaler.fit(target.reshape(-1, 1))
        self._is_fitted = True
        return self
    
    def transform_features(self, features):
        return self.feature_scaler.transform(features)
    
    def transform_target(self, target):
        return self.target_scaler.transform(target.reshape(-1, 1)).flatten()
    
    def inverse_transform_target(self, target):
        return self.target_scaler.inverse_transform(target.reshape(-1, 1)).flatten()
    
    def get_params(self):
        return {
            'feature_min': self.feature_scaler.data_min_,
            'feature_max': self.feature_scaler.data_max_,
            'target_min': self.target_scaler.data_min_,
            'target_max': self.target_scaler.data_max_,
        }

normalizer = Normalizer()

# ===== SEQUENCE CREATION =====
def create_sequences(features, target, seq_length, forecast_horizon=1):
    X, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon + 1):
        X.append(features[i : i + seq_length])
        y.append(target[i + seq_length + forecast_horizon - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ===== BUILD SEQUENCES PER GROUP =====
def build_sequences_per_group(df_clean, group_col, feature_cols, target_col,
                               normalizer, seq_length, forecast_horizon,
                               train_ratio, val_ratio):
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    groups = df_clean[group_col].unique() if group_col else ['__all__']
    
    # First pass: collect training data to fit normalizer
    train_features_all, train_target_all = [], []
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            continue
        train_end = int(n * train_ratio)
        train_features_all.append(grp_df[feature_cols].values[:train_end])
        train_target_all.append(grp_df[target_col].values[:train_end])
    
    normalizer.fit(np.concatenate(train_features_all), np.concatenate(train_target_all))
    print(f"Normalizer fitted on {np.concatenate(train_features_all).shape[0]} training samples.")
    
    # Second pass: normalize and create sequences
    skipped = 0
    for grp in groups:
        grp_df = df_clean[df_clean[group_col] == grp].sort_values(date_col) if group_col else df_clean.copy()
        n = len(grp_df)
        if n < seq_length + forecast_horizon + 10:
            skipped += 1
            continue
        
        features_norm = normalizer.transform_features(grp_df[feature_cols].values)
        target_norm = normalizer.transform_target(grp_df[target_col].values)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        for split, start, end, storage_X, storage_y in [
            ('train', 0, train_end, all_train_X, all_train_y),
            ('val', train_end - seq_length, val_end, all_val_X, all_val_y),
            ('test', val_end - seq_length, n, all_test_X, all_test_y),
        ]:
            feat = features_norm[start:end]
            tgt = target_norm[start:end]
            if len(feat) >= seq_length + forecast_horizon:
                X, y = create_sequences(feat, tgt, seq_length, forecast_horizon)
                storage_X.append(X)
                storage_y.append(y)
    
    print(f"Processed {len(groups) - skipped} groups, skipped {skipped}.")
    
    return (np.concatenate(all_train_X), np.concatenate(all_train_y),
            np.concatenate(all_val_X), np.concatenate(all_val_y),
            np.concatenate(all_test_X), np.concatenate(all_test_y))

train_X, train_y, val_X, val_y, test_X, test_y = build_sequences_per_group(
    df_clean, group_col_for_impute, FEATURE_COLS, TARGET_COL,
    normalizer, SEQUENCE_LENGTH, FORECAST_HORIZON, TRAIN_RATIO, VAL_RATIO
)

print(f"\n{'='*60}")
print(f"Train X: {train_X.shape}  |  Train y: {train_y.shape}")
print(f"Val   X: {val_X.shape}  |  Val   y: {val_y.shape}")
print(f"Test  X: {test_X.shape}  |  Test  y: {test_y.shape}")
````

**Cell 15 — PyTorch DataLoaders**

````python
// filepath: c:\Users\abhia\Documents\counterfactual_basis_kernel-main\notebooks\01_AQI_INDIA\01_AQI_India_EDA.ipynb
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train
```


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_classif
from pandas.api import types as ptypes
import warnings
warnings.filterwarnings("ignore")


def clean_and_engineer(df,
                       target_col: str | None = None,
                       null_threshold: float = 0.6,
                       zscore_threshold: float = 3.0,
                       top_k_features: int = 10,
                       verbose: bool = True):
    """
    Clean and feature-engineer a pandas DataFrame.

    Args:
        df: pandas.DataFrame (input, will be copied; original not modified)
        target_col: optional name of the target column (used for feature selection)
        null_threshold: fraction of missing values above which a column is dropped
        zscore_threshold: z-score cutoff to treat as outlier (applied across numeric columns)
        top_k_features: number of features to return in feature selection (if target provided)
        verbose: if True prints progress / summaries

    Returns:
        cleaned_df: pandas.DataFrame (cleaned & engineered)
        metadata: dict with keys:
            - original_columns (list)
            - removed_columns (list)
            - selected_features (list, possibly empty)
            - ollama_prompt (str)
    """
    # Work on a copy
    df = df.copy()
    original_columns = df.columns.tolist()
    removed_columns = []

    if verbose:
        print("Starting cleaning & feature engineering...")
        print(f"Original shape: {df.shape}")

    # -----------------------
    # 1) Safe ID-like detection (heuristic + uniq ratio)
    # -----------------------
    # substring heuristic
    substr_hits = [col for col in df.columns if any(s in col.lower() for s in ('id', 'serial', 'phone', 'number'))]
    # uniqueness heuristic: columns with uniqueness ratio > 0.95 are likely identifiers
    uniq_ratio = df.nunique(dropna=False) / len(df)
    uniq_hits = uniq_ratio[uniq_ratio > 0.95].index.tolist()
    id_candidates = sorted(set(substr_hits + uniq_hits))

    # only mark as removed_candidates (we'll drop them)
    if id_candidates:
        removed_columns.extend(id_candidates)
        df.drop(columns=id_candidates, inplace=True, errors='ignore')
        if verbose:
            print(f"Dropped id-like/unique columns: {id_candidates}")

    # -----------------------
    # 2) Drop zero-variance columns (<=1 unique value)
    # -----------------------
    zero_var = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    if zero_var:
        removed_columns.extend(zero_var)
        df.drop(columns=zero_var, inplace=True, errors='ignore')
        if verbose:
            print(f"Dropped zero-variance columns: {zero_var}")

    # -----------------------
    # 3) Drop high-null columns
    # -----------------------
    high_null = [col for col in df.columns if df[col].isnull().mean() > null_threshold]
    if high_null:
        removed_columns.extend(high_null)
        df.drop(columns=high_null, inplace=True, errors='ignore')
        if verbose:
            print(f"Dropped high-null (> {null_threshold*100:.0f}%) columns: {high_null}")

    # -----------------------
    # 4) Feature engineering - Dates
    # -----------------------
    date_cols = [col for col in df.columns if 'date' in col.lower() or ptypes.is_datetime64_any_dtype(df[col])]
    # attempt to convert string date-like columns
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            # leave as-is if conversion fails
            continue

    # For all datetime columns, extract parts and (optionally) drop original datetime
    datetime_cols = [col for col in df.columns if ptypes.is_datetime64_any_dtype(df[col])]
    for col in datetime_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        # keep original only if you really need it; we drop to follow your original flow
        df.drop(columns=[col], inplace=True, errors='ignore')
    if verbose and datetime_cols:
        print(f"Extracted date parts from: {datetime_cols}")

    # -----------------------
    # 5) Numeric transforms (log for skewed positive columns)
    # -----------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        # require at least 2 non-null values to compute skew
        ser = df[col].dropna()
        if ser.shape[0] < 2:
            continue
        # only if all positive or zeros, and skew high
        try:
            skew_val = ser.skew()
        except Exception:
            skew_val = 0
        if (ser >= 0).all() and skew_val > 1:
            # apply safe log1p
            df[col] = np.log1p(df[col])
            if verbose:
                print(f"Applied log1p to skewed column: {col} (skew={skew_val:.2f})")

    # -----------------------
    # 6) Categorical encoding (one-hot for low-cardinality)
    # -----------------------
    # choose object & category columns
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns.tolist()]
    # To avoid exploding dimensionality, only one-hot encode columns with cardinality <= 20 by default
    one_hot_cols = [c for c in cat_cols if df[c].nunique(dropna=False) <= 20]
    other_cat_cols = [c for c in cat_cols if c not in one_hot_cols]

    if one_hot_cols:
        df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)
        if verbose:
            print(f"One-hot encoded low-cardinality categorical columns: {one_hot_cols}")

    # For high-cardinality categorical columns, convert to frequency encoding (simple & safe)
    for c in other_cat_cols:
        freq = df[c].value_counts(dropna=False, normalize=True)
        df[c + "_freq_enc"] = df[c].map(freq).astype(float)
        df.drop(columns=[c], inplace=True, errors='ignore')
        if verbose:
            print(f"Applied frequency encoding to high-card column: {c}")

    # -----------------------
    # 7) Missing value imputation
    # -----------------------
    for col in df.columns:
        if df[col].isnull().any():
            if ptypes.is_numeric_dtype(df[col]):
                # numeric -> median
                med = df[col].median()
                df[col].fillna(med, inplace=True)
            else:
                # categorical/others -> mode if exists else fill with a placeholder
                modes = df[col].mode()
                if not modes.empty:
                    df[col].fillna(modes[0], inplace=True)
                else:
                    df[col].fillna("missing", inplace=True)
    if verbose:
        print("Imputed missing values (median for numeric, mode/placeholder for others).")

    # -----------------------
    # 8) Remove duplicates
    # -----------------------
    before_dup = df.shape[0]
    df.drop_duplicates(inplace=True)
    if verbose:
        print(f"Removed duplicates: {before_dup - df.shape[0]} rows dropped.")

    # -----------------------
    # 9) Outlier removal (z-score across numeric columns) - non-cumulative safe approach
    # -----------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        # compute z-scores per numeric column; handle constant columns safely
        means = df[numeric_cols].mean()
        stds = df[numeric_cols].std(ddof=0).replace(0, np.nan)  # avoid div-by-zero
        z = (df[numeric_cols] - means) / stds
        # treat NaN z (from std==0) as 0 (not an outlier)
        z = z.fillna(0).abs()
        mask = (z < zscore_threshold).all(axis=1)  # keep rows that are not outliers in any numeric col
        before_out = df.shape[0]
        df = df.loc[mask].copy()
        if verbose:
            print(f"Outlier removal (z<{zscore_threshold}): {before_out - df.shape[0]} rows removed.")

    # -----------------------
    # 10) Optional feature selection (univariate)
    # -----------------------
    selected_features = []
    if target_col and target_col in df.columns:
        try:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            # choose scoring function
            if ptypes.is_numeric_dtype(y) and y.nunique() > 10:
                score_func = f_regression
            else:
                score_func = mutual_info_classif
            k = min(top_k_features, X.shape[1])
            selector = SelectKBest(score_func=score_func, k=k)
            selector.fit(X.fillna(0), y)  # ensure no NaNs for selector
            selected_features = X.columns[selector.get_support()].tolist()
            if verbose:
                print(f"Selected top-{k} features for target ({target_col}): {selected_features}")
        except Exception as e:
            if verbose:
                print("Feature selection skipped due to error:", str(e))

    else:
        if verbose:
            print("No target provided or target missing; skipping feature selection.")

    # # -----------------------
    # # 11) Prepare Ollama prompt (using original columns)
    # # -----------------------
    # ollama_prompt = (
    #     "Dataset context: (describe dataset briefly here)\n"
    #     f"Target column: {target_col}\n\n"
    #     f"Original columns: {original_columns}\n\n"
    #     "Task:\n"
    #     "1) Identify likely irrelevant/noisy features that should be removed (IDs, serial numbers, phone numbers, "
    #     "raw timestamps, constants, PII, or any column with no predictive value).\n"
    #     "2) For each suggested column, provide a 1-2 line reason explaining why.\n"
    #     "3) If a column could be transformed instead of dropped, suggest the transform.\n\n"
    #     "Return output as a Python list of column names to drop and a JSON mapping {column: reason}."
    # )

    # # final metadata
    # metadata = {
    #     "original_columns": original_columns,
    #     "removed_columns": removed_columns,
    #     "selected_features": selected_features,
    #     "ollama_prompt": ollama_prompt
    # }

    # if verbose:
    #     print("Cleaning & feature engineering complete.")
    #     print(f"Final shape: {df.shape}")
    print(df.head())
    return df

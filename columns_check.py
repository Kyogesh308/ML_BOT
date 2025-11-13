import pandas as pd

# Excel-style column generator
def generate_col_names(n):
    names = []
    for i in range(n):
        name = ""
        x = i
        while True:
            name = chr(ord('A') + (x % 26)) + name
            x = x // 26 - 1
            if x < 0:
                break
        names.append(name)
    return names


def load_and_fix_headers(dataset_path: str, file_type: int, target:str):
    """
    file_type:
        1 = CSV
        2 = Excel
        3 = JSON
        4 = Parquet
        5 = TSV/TXT
    """
     
    # -------- STEP 1 — Load with header detection --------
    if file_type == 1:
        df = pd.read_csv(dataset_path, header=0)
    elif file_type == 2:
        df = pd.read_excel(dataset_path, header=0)
    elif file_type == 3:
        df = pd.read_json(dataset_path)
    elif file_type == 4:
        df = pd.read_parquet(dataset_path)
    elif file_type == 5:
        df = pd.read_csv(dataset_path, sep="\t", header=0)
    else:
        raise ValueError("File type must be 1–5")
    

    original_columns = df.columns.tolist()

    # Detect "no header" or partial missing columns
    missing_cols = []

    for idx, col in enumerate(original_columns):
        if (
            col is None
            or str(col).strip() == ""
            or str(col).startswith("Unnamed")
        ):
            missing_cols.append(idx)

    # Case: All columns are numeric → probably no header
    if all(isinstance(col, int) for col in original_columns):
        df = pd.read_csv(dataset_path, header=None) if file_type in [1,5] else \
             pd.read_excel(dataset_path, header=None)

        total_cols = len(df.columns)
        new_cols = generate_col_names(total_cols)
        df.columns = new_cols
        print(f"⚠ No column headers detected → assigning: {new_cols}")
        return df

    # If there are SOME missing headers → fill those only
    if len(missing_cols) > 0:
        total_cols = len(df.columns)
        excel_names = generate_col_names(total_cols)

        new_columns = df.columns.tolist()
        print("\n⚠ Missing column names detected — fixing...")

        for idx in missing_cols:
            old_name = new_columns[idx]
            new_columns[idx] = excel_names[idx]
            print(f"→ Column {idx} renamed to '{new_columns[idx]}' (was '{old_name}')")

        df.columns = new_columns
        print("\n✅ Partial column-name fixing completed.")
        return df

    # Otherwise → everything fine
    print("✅ All column names are present. No changes made.")
    return df

import pandas as pd

def load_data(file_path="final_processed_fitbit.csv", target_column="is_sedentary"):
    """
    Loads and auto-splits the Fitbit dataset into features and target.

    Parameters:
        file_path (str): Path to the preprocessed CSV file.
        target_column (str): Name of the label column to predict.

    Returns:
        X (pd.DataFrame): Numeric feature columns only, excluding ID and date.
        y (pd.Series): Target column.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found. Check the path.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Drop label, ID/date, and any leakage features
    leakage_features = ['SedentaryMinutes']  # direct leakage
    exclude_cols = ['Id', 'ActivityDate'] + leakage_features + [target_column]
   
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Drop only if present
    feature_cols = numeric_cols.drop([col for col in exclude_cols + [target_column] if col in numeric_cols])

    print(print("Selected feature columns:", feature_cols.tolist()))

    # Create features (X) and labels (y)
    X = df[feature_cols]
    y = df[target_column]

    return X, y

#load_data()
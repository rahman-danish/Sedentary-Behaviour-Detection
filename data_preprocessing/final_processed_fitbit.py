import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the merged files
df_daily = pd.read_csv("merged_dailyActivity.csv")
df_sleep = pd.read_csv("merged_sleepDay.csv")
df_heartrate = pd.read_csv("merged_heartrate.csv")
df_steps = pd.read_csv("merged_minuteSteps.csv")

# Step 2: Convert date columns to datetime (to python datetime format)
# Explicit format=... is used for sleep and steps to avoid parsing issues

df_daily['ActivityDate'] = pd.to_datetime(df_daily['ActivityDate'])
df_sleep['SleepDay'] = pd.to_datetime(df_sleep['SleepDay'], format="%m/%d/%Y %I:%M:%S %p")
df_heartrate['Time'] = pd.to_datetime(df_heartrate['Time'],)
df_steps['ActivityMinute'] = pd.to_datetime(df_steps['ActivityMinute'], format="%m/%d/%Y %I:%M:%S %p")

# Step 3: Merge daily activity with sleep data on Id + Date
df_merged = pd.merge(df_daily, df_sleep, left_on=['Id', 'ActivityDate'],
                     right_on=['Id', 'SleepDay'], how='left')

# Step 4: Aggregate heartrate and steps (average per day)
df_heartrate['Date'] = df_heartrate['Time'].dt.date
df_steps['Date'] = df_steps['ActivityMinute'].dt.date

# Daily average heartrate and steps
hr_daily = df_heartrate.groupby(['Id', 'Date'])['Value'].mean().reset_index().rename(columns={'Value': 'AvgHeartRate'})
steps_daily = df_steps.groupby(['Id', 'Date'])['Steps'].mean().reset_index().rename(columns={'Steps': 'AvgStepsPerMinute'})

# Merge into main DataFrame
df_merged['Date'] = df_merged['ActivityDate'].dt.date
df_merged = pd.merge(df_merged, hr_daily, on=['Id', 'Date'], how='left')
df_merged = pd.merge(df_merged, steps_daily, on=['Id', 'Date'], how='left')

# Step 5: Drop unused or redundant columns
drop_cols = ['SleepDay', 'Date']
df_merged.drop(columns=drop_cols, inplace=True)

# Step 6: Handle missing values (optional - here we fill with 0)
df_merged.fillna(0, inplace=True)

# Add sedentary behavior label (1 = sedentary, 0 = active)
df_merged['is_sedentary'] = (df_merged['SedentaryMinutes'] > 600).astype(int)

# Step 7: Normalize numerical features
numeric_cols = df_merged.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
df_merged[numeric_cols] = scaler.fit_transform(df_merged[numeric_cols])

# Step 8: Save final processed dataset
df_merged.to_csv("final_processed_fitbit.csv", index=False)
print("✅ Final dataset saved to final_processed_fitbit.csv")

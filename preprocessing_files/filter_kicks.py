import pandas as pd

# Input and output file paths
inference_csv = "./inference_result/predict_baidu_window_15_noalign.csv"  # Replace with actual path if different
output_kicks_csv = "./inference_result/kicks_extracted_noalign.csv"
output_other_csv = "./inference_result/non_kicks_extracted_noalign.csv"

# Load the inference results
df = pd.read_csv(inference_csv)

# Define filtering conditions
kick_condition = (
    df["type"].str.contains("corner", case=False, na=False) |
    df["anonymized"].str.contains("free kick", case=False, na=False)
)

# Filter for corner and free kicks
df_kicks = df[kick_condition]

# Filter for all other events
df_non_kicks = df[~kick_condition]

# Save the filtered data
df_kicks.to_csv(output_kicks_csv, index=False)
df_non_kicks.to_csv(output_other_csv, index=False)

print(f"Corner/Free Kick events saved to {output_kicks_csv}. Total: {len(df_kicks)}")
print(f"Other events saved to {output_other_csv}. Total: {len(df_non_kicks)}")

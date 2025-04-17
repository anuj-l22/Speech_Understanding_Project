import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import sys

# --- Step 1: Load the Excel data ---
excel_path = "audiomarkdata_20k.csv"  # Update with the actual path to your Excel file
df = pd.read_csv(excel_path)

df['name'] = df['name'].str.replace(r'\.wav$', '', regex=True)

print(df.head())

# --- Step 2: Create a composite stratification key ---
# This key combines age, gender, and language.
df['group'] = df['age'].astype(str) + "_" + df['gender'].astype(str) + "_" + df['language'].astype(str)

# --- Step 3: Uniformly sample 5 samples per group ---
sampled_groups = []
for group_value, group_df in df.groupby('group'):
    # As we assume each group has at least 5 entries, we simply sample 5 rows.
    sampled_group = group_df.sample(n=5, random_state=42)
    sampled_groups.append(sampled_group)

# Combine all groups; total samples will be 200 groups * 5 = 1000 samples.
sampled_df = pd.concat(sampled_groups).reset_index(drop=True)
print("Total samples selected:", len(sampled_df))  # Should print 1000

# --- Step 4: Split the samples into training and testing sets ---
# We split the 1000 samples into 800 training and 200 testing samples.
train_df, test_df = train_test_split(sampled_df, test_size=200, random_state=42)
print("Training samples:", len(train_df), "Test samples:", len(test_df))

# --- Step 5: Copy the corresponding audio files based on the 'name' column ---
# Specify the directories for the source audio files and the destination folders.
source_audio_dir = "audiomark"     # Update with your audio files directory path
train_dest_dir = "audiomarkdata_audioseal_train"        # Update with your train folder path
test_dest_dir = "audiomarkdata_audioseal_test"          # Update with your test folder path

# Create destination directories if they don't exist
os.makedirs(train_dest_dir, exist_ok=True)
os.makedirs(test_dest_dir, exist_ok=True)

# Define the audio file extension (e.g., ".wav")
file_extension = ".wav"

def copy_audio_files(df_subset, destination_folder):
    for _, row in df_subset.iterrows():
        # Construct the full file name based on the 'name' column
        file_name = str(row['name']) 
        src_file = os.path.join(source_audio_dir, file_name)
        dest_file = os.path.join(destination_folder, file_name)
        # Check if the file exists before copying
        if os.path.exists(src_file):
            shutil.copy(src_file, dest_file)
        else:
            print(f"Warning: {src_file} does not exist.")

# Copy audio files for training and testing sets
copy_audio_files(train_df, train_dest_dir)
copy_audio_files(test_df, test_dest_dir)

print("Audio file extraction and splitting completed!")

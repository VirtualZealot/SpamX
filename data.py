import os
import pandas as pd
from utils import parse_message, extract_message_features
from glob import glob
import config as cfg

# Process the SpamAssassin dataset
def process_spamassassin_data(datasets, standard_words):

    # Store processed datasets as a list of dataframes
    processed_datasets = []

    # For each spamassassin csv, read the contents and process them into a dataframe
    for dataset_name, label in datasets.items():
        # Generate paths
        dataset_path = os.path.join(cfg.SPAMASSASSIN_DATASET_PATH, dataset_name)
        processed_path = os.path.join(cfg.SPAMASSASSIN_PROCESSED_PATH, f'processed_{dataset_name}.csv')

        # Process the dataset
        if os.path.isfile(processed_path) and os.path.getsize(processed_path) > 0:
            print(f"Processed file for {dataset_name} found at {processed_path}. Loading...")
            df = pd.read_csv(processed_path)
        else:
            print(f"Processing original dataset from {dataset_name}...")
            message_contents = []
            for file_path in glob(os.path.join(dataset_path, '*')):
                message_content = parse_message(file_path)
                message_features = extract_message_features(message_content, cfg.feature_columns, label, standard_words)
                message_contents.append(message_features)

            df = pd.DataFrame(message_contents)
            df['spam'] = label
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            df.to_csv(processed_path, index=False)
            print(f"Processed data saved to {processed_path}")

        processed_datasets.append(df.reset_index(drop=True))

    return processed_datasets


# Process the other datasets (Kaggle/HuggingFace)
def process_csv_data(dataset_path, processed_path, standard_words):

    # Read the dataset csv file and process the data into a dataframe
    if os.path.isfile(processed_path):
        print(f"Processed file found at {processed_path}. Loading...")
        df = pd.read_csv(processed_path)
    else:
        print(f"Processing Kaggle dataset from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        subjects = df['email'].values
        labels = df['spam'].values

        message_contents = []
        for subject, label in zip(subjects, labels):
            message_features = extract_message_features(subject, cfg.feature_columns, label, standard_words)
            message_contents.append(message_features)

        df = pd.DataFrame(message_contents)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")

    return df.reset_index(drop=True)

def load_data(standard_words):
    print("\nProcessing spam/ham datasets...")

    # Store SpamAssassin data as multiple dataframes
    dfs_spamassassin = process_spamassassin_data(cfg.SPAMASSASSIN_DATASETS, standard_words)
    # Reset index of each dataframe in the list
    dfs_spamassassin = [df.reset_index(drop=True) for df in dfs_spamassassin]

    # Store Kaggle data as a dataframe
    df_kaggle = process_csv_data(cfg.KAGGLE_DATASET_PATH, cfg.KAGGLE_PROCESSED_PATH, standard_words)

    # Store HuggingFace data as a dataframe
    df_huggingface = process_csv_data(cfg.HUGGINGFACE_DATASET_PATH, cfg.HUGGINGFACE_PROCESSED_PATH, standard_words)

    # Combine and store the datasets as a single dataframe
    combined_df = pd.concat(dfs_spamassassin + [df_kaggle] + [df_huggingface], ignore_index=True)

    return combined_df


def print_dataset_info(dataset_name, file_paths):

    # Print the current datset name
    print(f"\nDataset Info: {dataset_name}")
    total_spam = 0
    total_ham = 0
    overall = 0

    # Count the amount of spam/ham samples and update the totals
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        spam_count = df[df['spam'] == 1].shape[0]
        ham_count = df[df['spam'] == 0].shape[0]
        total_spam += spam_count
        total_ham += ham_count
        overall += ham_count + spam_count
        print(f"{file_path}:\tTotal Samples: {len(df)} (Spam: {spam_count} | Ham: {ham_count})")
    print(f"Total {dataset_name} Samples: {overall} (Spam: {total_spam} | Ham: {total_ham})")
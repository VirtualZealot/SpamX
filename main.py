import time
import config as cfg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_standard_words, scale_features, train_split, evaluate_model
from data import load_data, print_dataset_info
from models import train_lr_model, train_rf_model, train_xgb_model

if __name__ == '__main__':
    # Load standard words list and data
    standard_words = load_standard_words(cfg.STANDARD_WORDS_PATH)
    df = load_data(standard_words)
    x, y = df.drop('spam', axis=1), df['spam']

    # Scale the input features
    x_scaled, scaler = scale_features(x)

    # Print dataset information
    for dataset_name, file_paths in cfg.DATASETS.items():
        print_dataset_info(dataset_name, file_paths)

    # Count ham and spam in the entire dataset
    total_spam = df['spam'].sum()  # Assuming spam is encoded as 1
    total_ham = len(df) - total_spam

    print(f"\nTotal Combined Samples: {len(df)} (Spam: {total_spam} | Ham: {total_ham})\n")

    # Define model indices
    lr_model_name = "LogisticRegression"
    xgb_model_name = "XGBoost"
    rf_model_name = "RandomForest"


    # Initialize 2 DataFrames to collect the average and full metrics for all trials
    average_results_df = pd.DataFrame()
    all_trial_results_df = pd.DataFrame()

    for split in cfg.test_splits:
        train_split_percentage = 1 - split
        split_results = []

        for trial in range(cfg.trials):
            # Split data into training and testing sets for each trial
            x_train, x_test, y_train, y_test = train_split(x_scaled, y, split, trial)
            num_training_parameters = len(y_train)
            num_testing_parameters = len(y_test)

            # Dictionary to hold the training time for each model
            training_times = {}

            # Train Logistic Regression and measure time
            start_time = time.time()
            lr_model = train_lr_model(x_train, y_train)
            training_times[lr_model_name] = time.time() - start_time

            # Train Random Forest and measure time
            start_time = time.time()
            rf_model = train_rf_model(x_train, y_train)
            training_times[rf_model_name] = time.time() - start_time

            # Train XGBoost and measure time
            start_time = time.time()
            xgb_model = train_xgb_model(x_train, y_train)
            training_times[xgb_model_name] = time.time() - start_time

            # Evaluate models, store the results and print the results to the console
            for model, model_name in [(lr_model, lr_model_name), (rf_model, rf_model_name), (xgb_model, xgb_model_name)]:
                accuracy, precision, recall, f1 = evaluate_model(model, x_test, y_test)
                trial_results = {
                    'Train|Test %': train_split_percentage,
                    'Training Samples': len(x_train),
                    'Testing Samples': len(x_test),
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'Training Time (s)': training_times[model_name]  # Add training time here
                }
                split_results.append(trial_results)

        # Create a DataFrame from the results of all trials for the current split
        split_df = pd.DataFrame(split_results)
        all_trial_results_df = pd.concat([all_trial_results_df, split_df], ignore_index=True)

        # Compute the mean of the metrics across all trials for the current split, including training time
        mean_results = split_df.groupby(
            ['Train|Test %', 'Training Samples', 'Testing Samples', 'Model']).mean().reset_index()

        # Append mean results to the average_results_df
        average_results_df = pd.concat([average_results_df, mean_results], ignore_index=True)

    # Print the average results table
    print(average_results_df.to_string(index=False))

    # Prepare a 2x2 grid of subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for i, metric in enumerate(metrics):
        # Create a boxplot for each model
        sns.boxplot(x='Training Samples', y=metric, hue='Model', data=all_trial_results_df, ax=axs[i // 2, i % 2],
                    palette='Set2')

        # Set titles with bold font
        axs[i // 2, i % 2].set_title(f'{metric}',
                                     fontdict={'fontsize': 15, 'fontweight': 'bold'})

        # Set axis labels with italic font
        axs[i // 2, i % 2].set_xlabel('# Of Training Samples', fontdict={'style': 'italic'}, fontsize=12)
        axs[i // 2, i % 2].set_ylabel(metric, fontdict={'style': 'italic'}, fontsize=12)

        axs[i // 2, i % 2].legend(title='Model')

    # Name the entire figure with bold title
    fig.suptitle(f'Performance Metrics By Training Size (Trials: {cfg.trials}, Total Samples: {len(df)})', fontweight='bold', fontsize=18)

    # Adjust layout for readability
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Show plot
    plt.show()
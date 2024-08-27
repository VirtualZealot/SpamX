import os

# Testing Parameters
trials = 10
test_splits = \
    [0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01]

# LogisticRegression Parameters
lr_params = {
    'C': 1.0,
    'solver': 'liblinear',
    'penalty': 'l2',
    'max_iter': 100
}

# RandomForest Parameters
rf_params = {
    'n_estimators': 50,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}

# XgBoost Parameters
xgb_params = {
    'n_estimators': 50,
    'max_depth': 10,
    'learning_rate': 0.1,
    'subsample': 1,
    'colsample_bytree': 1,
    'eval_metric': 'logloss'
}

# Define paths for the dataset directory
DATASET_DIR = 'Datasets'
ORIGINAL_DATASET_DIR = os.path.join(DATASET_DIR, 'Original')
PROCESSED_DATASET_DIR = os.path.join(DATASET_DIR, 'Processed')

# Define path for standard words file
STANDARD_WORDS_PATH = os.path.join(DATASET_DIR, 'Words', 'standard_english_words')

# Define paths for SpamAssassin dataset
SPAMASSASSIN_DATASET_PATH = os.path.join(ORIGINAL_DATASET_DIR, 'SpamAssassin')
SPAMASSASSIN_PROCESSED_PATH = os.path.join(PROCESSED_DATASET_DIR, 'SpamAssassin')
SPAMASSASSIN_DATASETS = {'spam_1': 1, 'spam_2': 1, 'easy_ham_1': 0, 'easy_ham_2': 0, 'hard_ham': 0}

# Define paths for Kaggle dataset
KAGGLE_DATASET_PATH = os.path.join(ORIGINAL_DATASET_DIR, 'Kaggle', 'kaggle_spam_ham.csv')
KAGGLE_PROCESSED_PATH = os.path.join(PROCESSED_DATASET_DIR, 'Kaggle', 'processed_kaggle.csv')

# Define paths for HuggingFace dataset
HUGGINGFACE_DATASET_PATH = os.path.join(ORIGINAL_DATASET_DIR, 'HuggingFace', 'huggingface_spam_ham.csv')
HUGGINGFACE_PROCESSED_PATH = os.path.join(PROCESSED_DATASET_DIR, 'HuggingFace', 'processed_huggingface.csv')

# Create a dictionary for the dataset paths
DATASETS = {
    'HuggingFace': [os.path.join(PROCESSED_DATASET_DIR, 'HuggingFace', 'processed_huggingface.csv')],
    'Kaggle': [os.path.join(PROCESSED_DATASET_DIR, 'Kaggle', 'processed_kaggle.csv')],
    'SpamAssassin': [
            os.path.join(PROCESSED_DATASET_DIR, 'SpamAssassin', 'processed_easy_ham_1.csv'),
            os.path.join(PROCESSED_DATASET_DIR, 'SpamAssassin', 'processed_easy_ham_2.csv'),
            os.path.join(PROCESSED_DATASET_DIR, 'SpamAssassin', 'processed_hard_ham.csv'),
            os.path.join(PROCESSED_DATASET_DIR, 'SpamAssassin', 'processed_spam_1.csv'),
            os.path.join(PROCESSED_DATASET_DIR, 'SpamAssassin', 'processed_spam_2.csv')
        ]
}

# Create a list of the features to be extracted
feature_columns = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_our', 'word_freq_guarantee'
        'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail',
        'word_freq_receive', 'word_freq_return', 'word_freq_will', 'word_freq_people', 'word_freq_report',
        'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you',
        'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_special',
        'word_freq_weightloss', 'word_freq_diet', 'word_freq_dieting', 'word_freq_powerful', 'word_freq_available',
        'word_freq_abdominal', 'word_freq_fat', 'word_freq_calorie', 'word_freq_link', 'word_freq_url',
        'word_freq_2000', 'word_freq_your', 'word_freq_mortgage', 'word_freq_buy', 'word_freq_prescription',
        'word_freq_pill', 'word_freq_pills', 'word_freq_save', 'word_freq_cheap', 'word_freq_cheapest',
        'word_freq_prozac', 'word_freq_cialis', 'word_freq_viagra', 'word_freq_valium', 'word_freq_enhancement',
        'word_freq_enhance', 'word_freq_orgasm', 'word_freq_orgasms', 'word_freq_sex', 'word_freq_sexy',
        'word_freq_sexual', 'word_freq_sluts', 'word_freq_naked', 'word_freq_erection', 'word_freq_miracle',
        'word_freq_pornography', 'word_freq_porn', 'word_freq_xxx', 'word_freq_money', 'word_freq_photoshop',
        'word_freq_microsoft', 'word_freq_software', 'word_freq_spyware', 'word_freq_spam', 'word_freq_rx',
        'word_freq_click', 'word_freq_unlimited', 'word_freq_browser', 'word_freq_subscribe', 'word_freq_trial',
        'word_freq_program', 'word_freq_windows', 'word_freq_xp', 'word_freq_professional', 'word_freq_nt',
        'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_lab', 'word_freq_857', 'word_freq_data',
        'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_30',
        'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re',
        'word_freq_table', 'word_freq_conference', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
        'phrase_freq_click_below', 'phrase_freq_click_the_link', 'phrase_freq_male_enhancement', 'phrase_freq_30_day',
        'phrase_freq_weight_loss',  'phrase_freq_all_natural', 'phrase_freq_body_weight',
        'phrase_freq_lose_weight', 'phrase_freq_no_prescription', 'phrase_freq_without_prescription',
        'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'char_freq_?',
        'non_standard_word_count', 'special_char_to_alpha_ratio', 'unusual_char_count', 'ascii_control_and_extended_count',
        'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'link_count', 'obfuscated_link_count',
        'spam'
    ]
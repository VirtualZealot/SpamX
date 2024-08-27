import regex as re
import email
import email.policy
from html import unescape
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_standard_words(file_path):
    standard_words = set()
    print("\nCreating dictionary of standard english words...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Convert to lowercase before adding to the set
                standard_words.add(line.strip().lower())
            print("Dictionary creation successful")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return standard_words

def parse_message(file_path):

    # For email messages, parse the content and cleanup the data
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_email = f.read()
    parsed_email = email.message_from_string(raw_email, policy=email.policy.default)

    message_content = ''
    for part in parsed_email.walk():
        if part.get_content_type() in ['text/plain', 'text/html']:
            try:
                part_payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or 'utf-8'
                part_text = part_payload.decode(charset, errors='ignore')

                # If it's HTML, strip HTML tags and decode HTML entities
                if part.get_content_type() == 'text/html':
                    part_text = re.sub(r'<[^>]+>', '', part_text)  # Strip HTML tags
                    part_text = unescape(part_text)  # Decode HTML entities

                message_content += part_text + ' '  # Add a space to separate parts
            except (AttributeError, LookupError, ValueError):
                continue

    return message_content.strip()  # Remove leading/trailing spaces


def extract_message_features(message_content, feature_columns, label, standard_words):
    # Initialize a dictionary to store the features
    features = dict.fromkeys(feature_columns, 0)  # Include 'spam' column for label

    # Calculate run lengths of uppercase letters
    capital_sequences = re.findall(r'([A-Z]+)', message_content)
    if capital_sequences:
        run_lengths = [len(seq) for seq in capital_sequences]
        features['capital_run_length_average'] = sum(run_lengths) / len(run_lengths)
        features['capital_run_length_longest'] = max(run_lengths)
        features['capital_run_length_total'] = sum(run_lengths)

    # Prepare message content for word and character frequency analysis by converting to lowercase
    message_content_lower = message_content.lower()

    # Match words with letters and numbers
    words = re.findall(r'\b\w+\b', message_content_lower)
    characters = re.findall(r'\S', message_content_lower)

    # Count the number of links
    links = re.findall(r'http[s]?://\S+|www\.\S+', message_content_lower)
    # Store the explicit link count
    features['link_count'] = len(links)

    # Normalize spaces around potential link fragments to simplify matching
    normalized_content = re.sub(r'\s+', ' ', message_content.lower())

    # Find all instances where 'http' or 'www' start a potential link
    potential_links_start = [match.start() for match in re.finditer(r'\b(http|www)\b', normalized_content)]

    # Search for any potential links that may be obfuscated
    obfuscated_links = []
    for start_index in potential_links_start:
        # Look ahead for a sequence that could be a domain name
        look_ahead = normalized_content[start_index:start_index + 100]
        potential_link_match = re.search(r'(http|www)\s?(:\s?/\s?/\s?)?\s?[\w.-]+\s?\.\s?(com|biz|org|net)\b', look_ahead)

        if potential_link_match:
            # Extract the matched string and remove all spaces
            matched_string = potential_link_match.group()
            cleaned_link = re.sub(r'\s+', '', matched_string)

            # Verify it forms a valid link after processing
            if re.match(r'(http|www)(://)?[\w.-]+\.(com|biz|org|net)', cleaned_link):
                obfuscated_links.append(cleaned_link)

    # Store the obfuscated link count
    features['obfuscated_link_count'] = len(obfuscated_links)

    # Detect non-standard words
    non_standard_words = set(words) - set(standard_words)
    features['non_standard_word_count'] = len(non_standard_words)

    # Compare ratio of special characters to alphanumeric characters
    special_chars = re.findall(r'[\W_]', message_content_lower)  # Include underscores in special characters
    alpha_num_chars = re.findall(r'[\w]', message_content_lower)  # Alphanumeric characters

    # Count unusual characters
    unusual_chars = re.findall(r'[{}[\]~|^]', message_content_lower)
    features['unusual_char_count'] = len(unusual_chars)

    # ASCII control (0-31 and 127) and extended (128-255) characters
    ascii_control_and_extended = re.findall(r'[\x00-\x1F\x7F-\xFF]', message_content)
    # Store the ascii character count
    features['ascii_control_and_extended_count'] = len(ascii_control_and_extended)

    # Store the ratio of special characters to alphanumeric characters and store the feature
    if alpha_num_chars:
        features['special_char_to_alpha_ratio'] = len(special_chars) / len(alpha_num_chars)
    else:
        features['special_char_to_alpha_ratio'] = 0

    # Calculate word frequencies as a percentage of total words and store the feature
    total_words = len(words)
    for word in words:
        if 'word_freq_' + word in features:
            features['word_freq_' + word] += (1.0 / total_words) * 100

    # Calculate character frequencies as a percentage of total characters and store the feature
    total_chars = len(characters)
    for char in characters:
        char_key = 'char_freq_' + char
        if char_key in features:
            features[char_key] += (1.0 / total_chars) * 100

    # Calculate phrase frequencies and store the features
    for phrase_key in feature_columns:
        if phrase_key.startswith('phrase_freq_'):
            # Extract the phrase from the feature name
            phrase = phrase_key[len('phrase_freq_'):].replace('_', ' ')
            phrase_count = len(re.findall(r'\b' + re.escape(phrase) + r'\b', message_content_lower))
            features[phrase_key] = (phrase_count / total_words) * 100 if total_words > 0 else 0

    # Assign the label
    features['spam'] = label

    return features

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def evaluate_model(model, x_test, y_test):

    # Store the model predictions for evaluation
    y_pred = model.predict(x_test)

    # Calculate the model performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1

def train_split(x_scaled, y, test_percentage, trial):
    # Train the model based on the scaled input values, the train|test split as a random trial
    return train_test_split(x_scaled, y, test_size=test_percentage, random_state=trial)
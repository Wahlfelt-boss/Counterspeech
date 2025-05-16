# Import necessary libraries first
import re
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Content,
    Part,
    HarmCategory,
    HarmBlockThreshold
)
import pandas as pd
import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io # For potential upload handling

# --- Configuration ---
PROJECT_ID = "gen-lang-client-0126734036" # Your Project ID
LOCATION = "us-central1"                 # Your Location
VERTEX_MODEL_NAME = "gemini-2.0-flash" # User requested model
DATA_FILE_NAME = '/content/2000_CS.csv'
CSV_DELIMITER = ';'
SAMPLE_SIZE = 2000 # Keep sample size small for debugging
RETRIES = 3
RETRY_DELAY = 5 # seconds
SLEEP_BETWEEN_CALLS = 1 # seconds (Implement this)
GENERATION_TEMPERATURE = 0.0  # deterministic
MAX_TOKENS = 15  # Should be enough for one category word

# Ensure categories are lowercase and sorted for consistency
COUNTERSPEECH_CATEGORIES = sorted([
    "humor", "facts", "question", "denouncing",
    "hypocrisy", "consequences", "positive",
    "affiliation", "culture", "negative", "other"
])
print(f"Using categories: {COUNTERSPEECH_CATEGORIES}")

# --- Define FEW_SHOT_EXAMPLES ---
# >>>>>>>>>>>> PASTE YOUR FEW_SHOT_EXAMPLES LIST HERE <<<<<<<<<<<<
# Example structure: FEW_SHOT_EXAMPLES = [{'hatespeech': '...', 'counterspeech': '...', 'category': 'facts'}, ...]
# Make sure the 'category' values here are lowercase and match COUNTERSPEECH_CATEGORIES
FEW_SHOT_EXAMPLES = [
     {'hatespeech': '"Muslims are invading our country!"', 'counterspeech': '"Immigration statistics show otherwise. The number of Muslim immigrants is much lower than often portrayed, and many contribute positively to the economy."', 'category': 'facts'},
     {'hatespeech': '"All feminists hate men."', 'counterspeech': '"That\'s a sweeping generalization. Feminism advocates for gender equality. Have you actually spoken to many feminists about their views?"', 'category': 'question'},
     {'hatespeech': '"Being gay is a choice and it\'s wrong."', 'counterspeech': '"Wow, telling people how to live based on assumptions? Sounds pretty controlling."', 'category': 'denouncing'},
     {'hatespeech': '"These people just need to work harder like the rest of us."', 'counterspeech': '"Says the person who inherited their wealth? Easy to talk about hard work from a position of privilege."', 'category': 'hypocrisy'},
     {'hatespeech': '"If you don\'t like it here, just leave!"', 'counterspeech': '"Leave? But I just ordered pizza! You gonna eat it all?"', 'category': 'humor'},
     {'hatespeech': '"Spreading lies about our group will not be tolerated!"', 'counterspeech': '"Making threats online can have serious legal consequences. You might want to rethink that."', 'category': 'consequences'},
]
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
if not FEW_SHOT_EXAMPLES:
    print("WARNING: FEW_SHOT_EXAMPLES list is empty. Classification might be poor.")

# --- Initialize Vertex AI ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel(VERTEX_MODEL_NAME)
    print(f"Vertex AI Initialized. Loaded model: {VERTEX_MODEL_NAME}")
except Exception as init_err:
    print(f"Error initializing Vertex AI or loading model: {init_err}")
    raise SystemExit("Vertex AI Initialization Failed")

# --- Prompt builder (Revised for Clarity) ---
def create_prompt(hs, cs, categories, examples):
    lines = [
        "Analyze the Hatespeech and Counterspeech pair provided.",
        "Your task is to classify the Counterspeech based on its primary strategy, choosing exactly one category from the list below.",
        "Categories: " + ", ".join(categories) + "\n",
        "Response Requirements: Respond with only the single category name in lowercase. Do not include punctuation, explanations, or any other text.\n",
        "--- Examples ---"
    ]
    for ex in examples:
        # Ensure example categories are lowercase
        ex_category = ex.get('category', 'other').lower()
        if ex_category not in categories:
             print(f"Warning: Few-shot example category '{ex_category}' not in defined categories. Defaulting to 'other' for example.")
             ex_category = 'other'
        lines.extend([
            f"Hatespeech: {ex.get('hatespeech', '')}",
            f"Counterspeech: {ex.get('counterspeech', '')}",
            f"Category: {ex_category}\n" # Use "Category:" consistently
        ])
    lines.extend([
        "--- Task ---",
        f"Hatespeech: {hs}",
        f"Counterspeech: {cs}",
        "Category:" # Prompt for the category
    ])
    return "\n".join(lines)

# --- Classification Function (Revised with Debugging and Better Parsing) ---
def classify_counterspeech(hs, cs, row_index): # Add index for debugging
    prompt = create_prompt(hs, cs, COUNTERSPEECH_CATEGORIES, FEW_SHOT_EXAMPLES)
    gen_cfg = GenerationConfig(temperature=GENERATION_TEMPERATURE, max_output_tokens=MAX_TOKENS)
    # Using BLOCK_ONLY_HIGH - consider BLOCK_NONE if safe content is blocked, but be cautious.
    safety = {cat: HarmBlockThreshold.BLOCK_ONLY_HIGH for cat in HarmCategory}

    print(f"\n--- Classifying Row Index: {row_index} ---") # DEBUG
    # print(f"DEBUG Prompt:\n{prompt}\n--------------------") # Uncomment to see full prompt if needed

    for attempt in range(RETRIES):
        try:
            resp = model.generate_content(
                contents=[Content(role="user", parts=[Part.from_text(prompt)])],
                generation_config=gen_cfg,
                safety_settings=safety,
                stream=False
            )

            # --- Robust Response Handling ---
            if not resp.candidates:
                finish_reason = getattr(resp, '_raw_response', {}).get('candidates', [{}])[0].get('finishReason', 'UNKNOWN')
                print(f"DEBUG Index {row_index}: No candidates. Finish Reason: {finish_reason}")
                if finish_reason == 'SAFETY': return "blocked_safety"
                # If blocked for other reasons or unknown, retry or fail
                if attempt + 1 < RETRIES:
                    print(f"DEBUG Index {row_index}: Retrying due to no candidates (Attempt {attempt+1})...")
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    continue
                else: return "error_no_candidates"

            # Check if parts exist
            if not resp.candidates[0].content.parts:
                 print(f"DEBUG Index {row_index}: Response has no parts.")
                 # Retry or fail
                 if attempt + 1 < RETRIES:
                     print(f"DEBUG Index {row_index}: Retrying due to no parts (Attempt {attempt+1})...")
                     time.sleep(RETRY_DELAY * (2 ** attempt))
                     continue
                 else: return "error_empty_parts"

            # Extract and clean text
            raw = resp.candidates[0].content.parts[0].text.strip()
            print(f"DEBUG Index {row_index}: Raw model output: '{raw}'") # <<< KEY DEBUG PRINT

            if not raw: # Handle empty string response
                print(f"DEBUG Index {row_index}: Model returned empty string.")
                # Consider empty string a failure for classification
                if attempt + 1 < RETRIES:
                     print(f"DEBUG Index {row_index}: Retrying due to empty string (Attempt {attempt+1})...")
                     time.sleep(RETRY_DELAY * (2 ** attempt))
                     continue
                else: return "error_empty_string"

            # --- Simplified Cleaning ---
            # 1. Lowercase
            cleaned = raw.lower()
            # 2. Remove potential leading/trailing punctuation common in direct answers
            cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", cleaned)

            print(f"DEBUG Index {row_index}: Cleaned output attempt: '{cleaned}'")

            # 3. Check if the result is now a valid category
            if cleaned in COUNTERSPEECH_CATEGORIES:
                print(f"DEBUG Index {row_index}: SUCCESS - Parsed as '{cleaned}'")
                return cleaned
            else:
                # Try splitting and taking first word as a fallback (less reliable)
                try:
                    first_word = cleaned.split()[0]
                    first_word_cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", first_word)
                    print(f"DEBUG Index {row_index}: Fallback - First word cleaned: '{first_word_cleaned}'")
                    if first_word_cleaned in COUNTERSPEECH_CATEGORIES:
                         print(f"DEBUG Index {row_index}: SUCCESS (Fallback) - Parsed as '{first_word_cleaned}'")
                         return first_word_cleaned
                except IndexError:
                    pass # If split fails (e.g., empty string somehow got here)

                # If still not found after fallback
                print(f"DEBUG Index {row_index}: Failed to parse valid category from '{raw}'. Returning 'parse_fail'.")
                return "parse_fail" # Specific code for parsing failure

        except Exception as e:
            print(f"ERROR Index {row_index}: API call failed (Attempt {attempt + 1}/{RETRIES}): {e}")
            msg = str(e).lower()
            # Check for retryable errors
            if any(code in msg for code in ["429", "quota", "resource exhausted", "500", "503", "service unavailable", "internal server error"]):
                if attempt + 1 < RETRIES:
                    wait_time = RETRY_DELAY * (2 ** attempt)
                    print(f"DEBUG Index {row_index}: Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue # Go to next attempt
                else:
                    print(f"DEBUG Index {row_index}: Max retries reached for API error.")
                    return "api_error_retries" # Specific code
            else: # Non-retryable API error
                 print(f"DEBUG Index {row_index}: Non-retryable API error.")
                 return "api_error_non_retryable"

    # If loop finishes without returning (should only happen if retries exhausted on specific conditions)
    print(f"DEBUG Index {row_index}: Max retries exhausted without success. Defaulting to 'error_max_retries'.")
    return "error_max_retries"

# --- Load & Sample Data ---
df_full = None
try:
    print(f"\n--- Loading Data ---")
    df_full = pd.read_csv(DATA_FILE_NAME, delimiter=CSV_DELIMITER, on_bad_lines='skip')
    print(f"Loaded {len(df_full)} rows (after skipping bad lines).")
    # Select necessary columns and clean
    df_full = df_full[['hatespeech', 'counterspeech', 'target']].astype(str).fillna('')
    print("Selected columns and handled missing values.")
except FileNotFoundError:
    print(f"ERROR: File not found at {DATA_FILE_NAME}")
except Exception as load_err:
    print(f"ERROR loading or processing data: {load_err}")

df_sample = None
if df_full is not None:
     # Take first N rows
     print(f"\n--- Selecting First {SAMPLE_SIZE} Rows ---")
     if len(df_full) >= SAMPLE_SIZE:
        df_sample = df_full.head(SAMPLE_SIZE).copy()
     else:
        df_sample = df_full.copy()
     print(f"Selected {len(df_sample)} rows for processing.")
else:
    print("Cannot proceed without loaded data.")
    raise SystemExit("Data Loading Failed")


# --- Run Classification (with sleep and index passing) ---
predictions = []
print(f"\n--- Running Classification on {len(df_sample)} Rows ---")
start_cls_time = time.time()

# Use enumerate to get index easily
for index, row in df_sample.iterrows():
    if not row['hatespeech'] or not row['counterspeech']:
        result = 'skipped_empty'
        print(f"DEBUG Index {index}: Skipped due to empty input.")
    else:
        result = classify_counterspeech(row['hatespeech'], row['counterspeech'], index) # Pass index
        # Implement sleep *after* a successful or failed call, before the next one
        if index < len(df_sample) - 1: # Don't sleep after the last item
             print(f"DEBUG Index {index}: Sleeping for {SLEEP_BETWEEN_CALLS}s...")
             time.sleep(SLEEP_BETWEEN_CALLS)

    predictions.append(result)

end_cls_time = time.time()
print(f"\n--- Classification Finished ---")
print(f"Time taken: {end_cls_time - start_cls_time:.2f} seconds.")

# Add predictions to the DataFrame
df_sample['predicted_output'] = predictions

# --- Analyze Raw Predictions ---
print("\n--- Raw Prediction Results Summary ---")
print(df_sample['predicted_output'].value_counts())

# --- Evaluate ---
print("\n--- Evaluating Results ---")

# Define evaluation categories (all possible valid outcomes + error codes needing mapping)
eval_categories = COUNTERSPEECH_CATEGORIES
error_codes = ['parse_fail', 'api_error_retries', 'api_error_non_retryable',
               'error_max_retries', 'blocked_safety', 'error_no_candidates',
               'error_empty_parts', 'error_empty_string', 'skipped_empty']

# Map true labels
df_sample['true_eval'] = df_sample['target'].str.lower().apply(
    lambda x: x if x in eval_categories else 'other'
)
# Map predictions: Map specific error codes and unrecognized outputs to 'other'
# Keep valid categories as they are.
def map_prediction(pred):
    if pred in eval_categories:
        return pred
    else: 
        return 'other'

df_sample['pred_eval'] = df_sample['predicted_output'].apply(map_prediction)

print("\n--- Processed Labels for Evaluation Summary ---")
print("True labels distribution:\n", df_sample['true_eval'].value_counts())
print("\nMapped predictions distribution:\n", df_sample['pred_eval'].value_counts())

report_labels = sorted(list(set(df_sample['true_eval']) | set(df_sample['pred_eval'])))

print("\n--- Classification Report ---")
try:
    print(classification_report(df_sample['true_eval'], df_sample['pred_eval'], labels=report_labels, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(df_sample['true_eval'], df_sample['pred_eval']):.4f}")
except Exception as report_err:
     print(f"Could not generate classification report: {report_err}")
     print("Check if true/predicted labels contain unexpected values after processing.")


# --- Confusion Matrix ---
print("\n--- Confusion Matrix ---")
try:
    cm = confusion_matrix(df_sample['true_eval'], df_sample['pred_eval'], labels=report_labels)
    plt.figure(figsize=(max(8, len(report_labels)*0.8), max(6, len(report_labels)*0.6)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=report_labels, yticklabels=report_labels)
    plt.xlabel('Predicted Label (Mapped)')
    plt.ylabel('True Label (Mapped)')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
except Exception as plot_err:
    print(f"Could not generate confusion matrix plot: {plot_err}")


# Display some examples of failures
failures = df_sample[df_sample['true_eval'] != df_sample['pred_eval']].copy()
failures['raw_pred'] = failures['predicted_output'] # Show the raw output for failures
print(f"\n--- Sample of Misclassified/Failed Rows ({len(failures)} total) ---")
print(failures[['hatespeech', 'counterspeech', 'target', 'true_eval', 'raw_pred', 'pred_eval']].head(10))

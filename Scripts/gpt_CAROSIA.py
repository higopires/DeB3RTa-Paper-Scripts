import openai
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc
)
import numpy as np
from transformers import AutoTokenizer
import argparse

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge")

def classify_text(message):
    # Tokenize the text and limit to 128 tokens
    tokens = tokenizer(
        message,
        truncation=True,
        max_length=128,
        return_tensors='pt',
        add_special_tokens=False
    )
    # Decode the tokens back to string
    truncated_message = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    
    # Create the prompt with the truncated message
    prompt = f"""Classify the following text of news updates about the Brazilian financial market as either positive (indicating favorable market conditions) or negative (indicating unfavorable conditions).
    Message: '{truncated_message}'.
    The output should only contain two words: positive or negative."""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        # Also get the confidence score for PR-AUC calculation
        prediction = response.choices[0].message['content'].strip()
        confidence = 1.0 if prediction == 'positive' else 0.0
        return prediction, confidence
    except Exception as e:
        print(f"Error occurred during classification: {e}")
        return None, None

def calculate_metrics(true_labels, predictions, prediction_probas):
    """
    Calculate and return all relevant metrics including PR-AUC.
    """
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'f1_macro': f1_score(true_labels, predictions, average='macro'),
        'precision_macro': precision_score(true_labels, predictions, average='macro'),
        'recall_macro': recall_score(true_labels, predictions, average='macro')
    }
    
    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(true_labels, prediction_probas)
    metrics['pr_auc'] = auc(recall, precision)
    
    return metrics

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Text classification script')
    parser.add_argument('--folder', type=str, required=True, help='Folder containing test.csv')
    args = parser.parse_args()
    
    # Load and preprocess the dataset
    dataset_path = os.path.join(args.folder, 'test.csv')
    dataset = pd.read_csv(dataset_path)
    
    # Ensure columns exist
    required_columns = ['text', 'label']
    if not all(col in dataset.columns for col in required_columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    
    # Initialize lists to store predictions, probabilities, and true labels
    predictions = []
    prediction_probas = []
    true_labels = []
    
    # Perform classification
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        prediction, confidence = classify_text(row['text'])
        if prediction is not None:
            predictions.append(1 if prediction == 'positive' else 0)
            prediction_probas.append(confidence)
            true_labels.append(row['label'])
    
    # Check if there are any valid predictions and labels
    if len(predictions) == 0 or len(true_labels) == 0:
        print("No valid predictions or true labels to evaluate.")
        return
    
    # Convert lists to numpy arrays
    predictions = np.array(predictions)
    prediction_probas = np.array(prediction_probas)
    true_labels = np.array(true_labels)
    
    # Calculate all metrics
    metrics = calculate_metrics(true_labels, predictions, prediction_probas)
    
    # Print results
    print("\nClassification Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-score Macro: {metrics['f1_macro']:.4f}")
    print(f"Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Recall Macro: {metrics['recall_macro']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")

if __name__ == "__main__":
    main()
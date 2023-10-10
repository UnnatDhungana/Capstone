# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:46:06 2023

@author: DhunganaBrothers
"""

# this program is for model architecture and data training
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.layers import Dropout
from fuzzywuzzy import fuzz
from Crypto.Cipher import AES

# import dataset
df = pd.read_csv('eula_dataset_without_stopwords.csv')

df['Clause']=df['Clause'].fillna('')
df['Label']=df['Label'].fillna('')

# Check for any unexpected data types in the 'Clause' and 'Label' columns
unexpected_clause_data = df[~df['Clause'].apply(lambda x: isinstance(x, str))]
unexpected_label_data = df[~df['Label'].apply(lambda x: isinstance(x, str))]

if not unexpected_clause_data.empty:
    print("Unexpected data in 'Clause' column:")
    print(unexpected_clause_data)

if not unexpected_label_data.empty:
    print("Unexpected data in 'Label' column:")
    print(unexpected_label_data)
# assign clause and labels

clause_text = df['Clause'].values

labels = df['Label'].values


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clause_text)
X = tokenizer.texts_to_sequences(clause_text)
X = pad_sequences(X, maxlen=300)

# Encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------------

# Defining model architecture for training
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
model.add(Conv1D(256, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(256, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(set(labels)), activation='softmax'))
# compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --------------------------------------------------------------------------------
# model training
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Function to extract text from a PDF document
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Example: Extract text from an input PDF agreement
input_pdf_file = 'EULA.pdf'  # Replace with your input PDF agreement path
input_text = extract_text_from_pdf(input_pdf_file)
 
# Define the delimiter based on your document's structure
clause_delimiter = 'Clause'  # Modify this to match the actual delimiter in your document

# Preprocess the input text
input_clauses = input_text.split(clause_delimiter)
input_sequence = tokenizer.texts_to_sequences([input_clauses])
input_sequence = pad_sequences(input_sequence, maxlen=X.shape[1])



# Predict the category for each clause in the input PDF document
predicted_labels = model.predict(input_sequence)

def print_predicted_categories(predicted_labels, label_encoder):
    """
    Print predicted categories.
    """
    predicted_labels = np.array(predicted_labels).reshape(-1)
    predicted_categories = label_encoder.inverse_transform(predicted_labels)
    for i, category in enumerate(predicted_categories):
        print(f"Clause {i+1}: {category}")
# Function to print the predicted categories
print_predicted_categories(predicted_labels, label_encoder)


# Print input_clauses and predicted_categories for debugging
print("Input Clauses:")
print(input_clauses)
print("Predicted Categories:")
# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clause_text)
X = tokenizer.texts_to_sequences(clause_text)
X = pad_sequences(X, maxlen=300)

# Encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------------

# Defining model architecture for training
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
model.add(Conv1D(256, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(256, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(set(labels)), activation='softmax'))
# compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --------------------------------------------------------------------------------
# model training
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Function to extract text from a PDF document
def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Example: Extract text from an input PDF agreement
input_pdf_file = 'EULA.pdf'  # Replace with your input PDF agreement path
input_text = extract_text_from_pdf(input_pdf_file)
 
# Define the delimiter based on your document's structure
clause_delimiter = 'Clause'  # Modify this to match the actual delimiter in your document

# Preprocess the input text
input_clauses = input_text.split(clause_delimiter)
input_sequence = tokenizer.texts_to_sequences([input_clauses])
input_sequence = pad_sequences(input_sequence, maxlen=X.shape[1])



# Predict the category for each clause in the input PDF document
predicted_labels = model.predict(input_sequence)
predicted_probabilities = model.predict_proba(input_sequence)
predicted_categories = label_encoder.inverse_transform(predicted_labels.reshape(-1))
def print_predicted_categories(predicted_labels, label_encoder):
    """
    Print predicted categories.
    """
    predicted_labels = np.array(predicted_labels).reshape(-1)
    predicted_categories = label_encoder.inverse_transform(predicted_labels)
    for i, category in enumerate(predicted_categories):
        print(f"Clause {i+1}: {category}")
# Function to print the predicted categories
print_predicted_categories(predicted_labels, label_encoder)


# Print input_clauses and predicted_categories for debugging
print("Input Clauses:")
print(input_clauses)
print("Predicted Categories:")

print(predicted_categories)

# Initialize a list to store the clauses and categories
extracted_data = []

# Iterate through the clauses and predicted categories
for clause, category in zip(input_clauses, predicted_categories):
    
    extracted_data.append(f"Clause: {clause.strip()}\nPredicted Category: {category}\n")

# Save the extracted clauses and categories to a text file
output_file_path = '10octExtracted.txt'  # Specify the desired output file path
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.writelines(extracted_data)

print(f"Extracted clauses and categories saved to '{output_file_path}'")

# Create a dictionary to store the actual categories from the dataset
actual_categories = dict(zip(df['Clause'], df['Label']))

# Display the predicted categories for each clause

for category in predicted_categories:
    print(f"Predicted Category: {category}")
    
# Initialize a list to store the clauses and categories
extracted_data = []

# Iterate through the clauses and predicted categories
for clause, category in zip(input_clauses, predicted_categories):
    
    extracted_data.append(f"Clause: {clause.strip()}\nPredicted Category: {category}\n")

# Save the extracted clauses and categories to a text file
output_file_path = '10octExtracted.txt'  # Specify the desired output file path
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.writelines(extracted_data)

print(f"Extracted clauses and categories saved to '{output_file_path}'")

# Create a dictionary to store the actual categories from the dataset
actual_categories = dict(zip(df['Clause'], df['Label']))

# Initialize lists to store True positives and False positives
true_positives = []
false_positives = []

# Define a similarity threshold for classification
similarity_threshold = 80  # Adjust as needed

# Compare the predicted categories with actual categories using fuzzy string matching
for extracted_item in extracted_data:
    # Split the extracted_item into two parts, if possible
    parts = extracted_item.split(':')
    
    if len(parts) == 2:
        clause, predicted_category = parts
        clause = clause.strip()
        predicted_category = predicted_category.strip()
        
        # Find the most similar clause from the dataset
        best_match = None
        best_similarity = 0
        
        for dataset_clause in df['Clause']:
            similarity_score = fuzz.token_sort_ratio(clause, dataset_clause)
            if similarity_score > best_similarity:
                best_similarity = similarity_score
                best_match = dataset_clause
        
        if best_similarity >= similarity_threshold and actual_categories.get(best_match) == predicted_category:
            true_positives.append(extracted_item)
        else:
            false_positives.append(extracted_item)
    else:
        # Handle cases where the extracted_item doesn't have the expected format
        print(f"Skipping invalid line")

# Save True positives and False positives to respective files
with open('Truepositive.txt', 'w', encoding='utf-8') as tp_file:
    tp_file.write('\n'.join(true_positives))

with open('Falsepositive.txt', 'w',encoding='utf-8') as fp_file:
    fp_file.write('\n'.join(false_positives))

print("True positives saved to 'Truepositive.txt'")
print("False positives saved to 'Falsepositive.txt'")
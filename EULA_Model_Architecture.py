# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:46:06 2023

@author: asimd
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
X = pad_sequences(X)

# Encoding labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------------------------

# Defining model architecture for training
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
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
input_pdf_file = 'macOSVentura.pdf'  # Replace with your input PDF agreement path
input_text = extract_text_from_pdf(input_pdf_file)

# Preprocess the input text
input_clauses = input_text.split('\n')
input_sequence = tokenizer.texts_to_sequences([input_clauses])
input_sequence = pad_sequences(input_sequence, maxlen=X.shape[1])

# Predict the category for each clause in the input PDF document
predicted_labels = model.predict(input_sequence)
predicted_categories = label_encoder.inverse_transform(np.argmax(predicted_labels, axis=1))

# Display the predicted categories for each clause
for category in predicted_categories:
    print(f"Predicted Category: {category}")
    
# Initialize a list to store the clauses and categories
extracted_data = []

# Iterate through the clauses and predicted categories
for clause, category in zip(input_clauses, predicted_categories):
    extracted_data.append(f"Clause: {clause.strip()}\nPredicted Category: {category}\n")

# Save the extracted clauses and categories to a text file
output_file_path = 'extracted_clauses.txt'  # Specify the desired output file path
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.writelines(extracted_data)

print(f"Extracted clauses and categories saved to '{output_file_path}'")


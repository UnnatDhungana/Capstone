# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:38:59 2023

@author: DhunganaBrothers
"""

# This takes the csv file and removes the stopwords from the dataset

import pandas as pd
import nltk
from nltk.corpus import stopwords

# Load your original dataset
df = pd.read_csv('dataset.csv')  # Replace with your dataset path

# Get the list of stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    # Check if the text is a string (not NaN)
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)
    else:
        return text  # Return the non-string value as is

# Apply the remove_stopwords function to the 'clause' column
df['Clause'] = df['Clause'].apply(remove_stopwords)

# Save the dataset with stopwords removed to a new CSV file
df.to_csv('eula_dataset_without_stopwords.csv', index=False)
print("Stop Words have been removed from the dataset")
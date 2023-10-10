# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:35:03 2023

@author: DhunganaBrothers
"""
# This program converts the clauses.txt file to a csv file.
import csv

# reading in the text file
input_file = 'clauses_data.txt'
output_file = 'dataset.csv'
delimiter = ','

with open(input_file, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()
    data = [line.strip().split(delimiter) for line in lines]

with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Clause', 'Label'])
    writer.writerows(data)
print(f'CSV file "{output_file}" has been created from "{input_file}" with delimiter "{delimiter}".')

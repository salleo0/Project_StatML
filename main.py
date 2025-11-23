import numpy as np
import pandas as pd


file_path = 'drug_consumption.data'

column_names = [
    'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
    'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
    'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
    'Nicotine', 'Semer', 'VSA'
]

data_frame = pd.read_csv(file_path, names=column_names, header=None)

# Semeron is a fictitious drug used to identify over-claimers
# We remove all instances where != CL0
data_frame = data_frame[data_frame['Semer'] == 'CL0']

print(data_frame)

# Crack small dataset
# Cocaine, cannabis, LSD larger datasets

drug_names = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
    'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
    'Nicotine', 'Semer', 'VSA'
]

for drug in drug_names:
    data_frame[drug] = data_frame[drug].apply(lambda x: 1 if x in ['CL3','CL4','CL5','CL6'] else 0)

print(data_frame)
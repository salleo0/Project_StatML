import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def import_dataset(file_path):
    
    column_names = [
        'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
        'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Choc', 'Coke', 'Crack',
        'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
        'Nicotine', 'Semeron', 'VSA'
    ]

    drug_names = [
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Choc', 'Coke', 'Crack',
        'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
        'Nicotine', 'Semeron', 'VSA'
    ]

    data_frame = pd.read_csv(file_path, names=column_names, header=None)

    # Semeron is a fictitious drug used to identify over-claimers
    # We remove all instances where Semeron != CL0 and then remove the column
    data_frame = data_frame[data_frame['Semeron'] == 'CL0']
    data_frame.drop('Semeron', axis=1, inplace=True)
    drug_names.remove('Semeron')

    # We simplify the dataframe into USER (1) and NON-USER (0)
    for drug in drug_names:
        data_frame[drug] = data_frame[drug].apply(lambda x: 1 if x in ['CL3','CL4','CL5','CL6'] else 0)

    return data_frame

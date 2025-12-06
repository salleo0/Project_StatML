import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def import_dataset(file_path):
    
    # FEATURES
    column_names = [
        'ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity',
        'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS',
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Choc', 'Coke', 'Crack',
        'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
        'Nicotine', 'Semeron', 'VSA'
    ]

    # OUTPUT
    drug_names = [
        'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caffeine', 'Cannabis', 'Choc', 'Coke', 'Crack',
        'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms',
        'Nicotine', 'Semeron', 'VSA'
    ]

    # IMPORTAZIONE DATASET
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

# Things to explain (mainly about the dataset) and the changes made on it:
#1. Goal of the main study based on this dataset 
#2. Input and output varibles
#3. The role of the output variable 'Semeron' and how it has been treated in the dataset we created
#4. The simplification of the categories of the output variables (the various drugs): the initial categories were splitted in two groups
#   and the problem became a classification problem
#5. The drop of the input variables 'Ethnicity' and 'Country'
#6. The choice of one specific/more specific (discuss about this) output variables. Why it has been made that particular choice?
#   The exploratory analysis could help to answer this question.

def feat_barplot(data_frame, x):
    # Relative frequencies for the input feature x
    freq = data_frame[x].value_counts(normalize=True)

    # Barplot with relative frequencies
    plt.figure(figsize=(8, 3))
    freq.plot(kind='bar')
    plt.xticks(rotation=90)
    plt.ylabel('Relative frequency')
    plt.title(f'Relative frequencies for {x}')
    plt.tight_layout()

    # Saving the plot
    filename = f'{x}_relative_barplot.png'
    folder = 'dataset_pictures'
    filepath = os.path.join(folder, filename)
    plt.savefig(os.path.join(folder, filename))
    plt.close()

    # Returning the relative frequencies in case I have to watch at the specific values
    return freq


def proportions_plot(data_frame, y):
    # Relative frequencies (0 and 1) for each drug
    users_per_drug_rel = {
        drug: data_frame[drug].value_counts(normalize=True).sort_index()
        for drug in y
    }

    users_rel_df = pd.DataFrame(users_per_drug_rel).fillna(0.0)

    # Barplot with the relative frequencies for each drug
    fig, ax = plt.subplots(figsize=(15, 8))
    users_rel_df.plot(kind='bar', width=0.8, ax=ax)
    ax.set_title('Relative frequency of users per drug type')
    ax.set_xlabel('User (1) and Non-User (0)')
    ax.set_ylabel('Proportion of people')
    ax.legend(title='Drugs', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    fig.tight_layout()

    # Saving the plot
    plt.savefig('dataset_pictures/relative_users_per_drug.png')
    plt.close(fig)

    # Returning the relative frequencies for each drug in case of need
    return users_rel_df
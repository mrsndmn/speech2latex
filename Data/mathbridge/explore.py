import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('MathBridge_train_cleaned_normalized.csv')

    for _, row in df[['equation', 'formula_normalized']].sample(100).iterrows():

        print('equation           ', row['equation'])
        print('formula_normalized ', row['formula_normalized'])
        print('-' * 100)
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = pd.read_csv('MathBridge_train_cleaned_normalized.csv')

    formulas = df['formula_normalized'].unique()
    train_formulas, test_formulas = train_test_split(formulas, test_size=0.1, random_state=42)

    df_train = df[df['formula_normalized'].isin(train_formulas)]
    df_test = df[df['formula_normalized'].isin(test_formulas)]

    print(f"Train size: {len(df_train)}")
    print(f"Test size: {len(df_test)}")

    train_path = 'MathBridge_train_cleaned_normalized_train.csv'
    test_path = 'MathBridge_train_cleaned_normalized_test.csv'

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print("Saved files:")
    print(f"{train_path}")
    print(f"{test_path}")
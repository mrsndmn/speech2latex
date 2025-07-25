import datasets
import pandas as pd
import sys


if __name__ == "__main__":

    csv_path = sys.argv[1]

    df = pd.read_csv(csv_path)

    df['latex_with_dollars'] = df['latex'].apply(lambda x: f'$ {x} $')

    csv_path = csv_path.replace('.csv', '_with_dollars.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")

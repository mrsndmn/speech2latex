import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--res_path', type=str, help="path to checkpoint")
args = parser.parse_args()


def check_latex_compilability(latex_code):
    try:
        # Create a dummy figure
        fig = plt.figure()
        plt.text(0.5, 0.5, f"${latex_code}$", fontsize=12, ha='center', va='center')
        
        # Save the figure to a temporary PDF file
        with PdfPages('/tmp/temp.pdf') as pdf:
            pdf.savefig(fig)
        
        plt.close(fig)
        return True  # LaTeX code compiled successfully
    except:
        plt.close(fig)
        return False  # LaTeX code failed to compile

# Sample DataFrame with a column of LaTeX code
df = pd.read_csv(args.res_path)
# Check each LaTeX code for compilability
df['compiles'] = df["latex_pred"].apply(check_latex_compilability)

# Display the DataFrame with compilation results
print(df['compiles'].sum() / df.shape[0])


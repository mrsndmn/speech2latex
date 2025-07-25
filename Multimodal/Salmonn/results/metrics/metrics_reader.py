import pandas as pd

path = "./metrics_1.csv"
df = pd.read_csv(path)

print("cer",df["cer"].mean() )
print("wer",df["wer"].mean())
print("rouge1",df["rouge1"].mean())
print("chrf",df["chrf"].mean())
print("chrfpp",df["chrfpp"].mean())
print("bleu",df["bleu"].mean())
print("sbleu",df["sbleu"].mean())
print("meteor",df["meteor"].mean())
print("cer_lower",df["cer_lower"].mean())
print("wer_lower",df["wer_lower"].mean())
print("rouge1_lower",df["rouge1_lower"].mean())
print("chrf_lower",df["chrf_lower"].mean())
print("chrfpp_lower",df["chrfpp_lower"].mean())
print("bleu_lower",df["bleu_lower"].mean())
print("sbleu_lower",df["sbleu_lower"].mean())
print("meteor_lower",df["meteor_lower"].mean())

import pandas as pd

res = pd.read_csv("processed/vocab_1000.tsv", sep="\t")
print(res)
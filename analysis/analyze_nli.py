import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bert_df = pd.read_csv("nlibias_results_bert.csv")
debiased_bert_df = pd.read_csv("nlibias_results_bert_debiased.csv")

# roberta_df = pd.read_csv("nlibias_results_roberta.csv")
# debiased_roberta_df = pd.read_csv("nlibias_results_roberta_debiased.csv")

# print(bert_df.GenderWord.unique())
# print(bert_df[bert_df["GenderWord"] == "gentleman"].groupby(["Prediction"]).count())
# print(
#     debiased_bert_df[debiased_bert_df["GenderWord"] == "gentleman"]
#     .groupby(["Prediction"])
#     .count()
# )


def pivot_dist(df):
    new_df = pd.DataFrame()
    for word in df.GenderWord.unique():
        new_df[word] = df[df["GenderWord"] == word].groupby(["Prediction"]).count()
    return new_df.T


bert_dist = pivot_dist(bert_df)
debiased_bert_dist = pivot_dist(debiased_bert_df)

width = 0.2
X_axis = np.arange(len(bert_dist.index))

plt.subplot(1, 2, 1)
plt.bar(X_axis - 0.2, bert_dist[0], width, label=0)
plt.bar(X_axis + 0, bert_dist[1], width, label=1)
plt.bar(X_axis + 0.2, bert_dist[2], width, label=2)
plt.xticks(X_axis, bert_dist.index)
plt.title("Original BERT")

plt.subplot(1, 2, 2)
plt.bar(X_axis - 0.2, debiased_bert_dist[0], width, label=0)
plt.bar(X_axis + 0, debiased_bert_dist[1], width, label=1)
plt.bar(X_axis + 0.2, debiased_bert_dist[2], width, label=2)
plt.xticks(X_axis, debiased_bert_dist.index)
plt.title("Debiased BERT")

plt.suptitle("Distribution of Predictions Across Gender Words for NLI-Bias")

plt.show()

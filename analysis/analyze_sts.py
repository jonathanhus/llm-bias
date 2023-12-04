import pandas as pd
import matplotlib.pyplot as plt

bert_df = pd.read_csv("stsbias_results_bert.csv")
debiased_bert_df = pd.read_csv("stsbias_results_bert_debiased.csv")

albert_df = pd.read_csv("stsbias_results_albert.csv")
debiased_albert_df = pd.read_csv("stsbias_results_albert_debiased.csv")

roberta_df = pd.read_csv("stsbias_results_roberta.csv")
debiased_roberta_df = pd.read_csv("stsbias_results_roberta_debiased.csv")


def find_means(df):
    return df.groupby(["Occupation"]).mean()


def find_gender_difference(df1, df2):
    df = pd.DataFrame([], index=df1.index)
    # df["Original"] = (df1["Male Score"] - df1["Female Score"]).abs()
    # df["Debiased"] = (df2["Male Score"] - df2["Female Score"]).abs()
    df["Percentage Change in Gender Bias Discrepancy"] = (
        df2["Male Score"] - df2["Female Score"]
    ).abs() / (df1["Male Score"] - df1["Female Score"]).abs() - 1
    return df


def find_means_and_difference_plot(df1, df2):
    df1_means = find_means(df1)
    df2_means = find_means(df2)
    all_means = find_gender_difference(df1_means, df2_means)
    ax = all_means.plot.bar(rot=90)
    ax.set_ylim([-1, 1])
    plt.xlabel("Occupation")
    plt.ylabel("Percentage Change in Gender Discrepancy (Limited to [-1, 1])")
    return ax


ax = find_means_and_difference_plot(bert_df, debiased_bert_df)
plt.title("BERT Percentage Change in Gender Discrepancy After Debiasing (STS-Bias)")
ax = find_means_and_difference_plot(albert_df, debiased_albert_df)
plt.title("ALBERT Percentage Change in Gender Discrepancy After Debiasing (STS-Bias)")
ax = find_means_and_difference_plot(roberta_df, debiased_roberta_df)
plt.title("RoBERTa Percentage Change in Gender Discrepancy After Debiasing (STS-Bias)")

plt.show()

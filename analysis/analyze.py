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
  df["Percentage Change in Gender Bias Discrepancy"] = (df2["Male Score"] - df2["Female Score"]).abs()/(df1["Male Score"] - df1["Female Score"]).abs() - 1
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

# bert_means = find_means(bert_df)
# debiased_bert_means = find_means(debiased_bert_df)
# all_bert_means = find_gender_difference(bert_means, debiased_bert_means)
# print(bert_means)
# print(debiased_bert_means)
# print(all_bert_means)
# all_bert_means.plot.bar(rot=90)
# plt.xticks(rotation=90)

# albert_means = find_means(albert_df)
# debiased_albert_means = find_means(debiased_albert_df)

# roberta_means = find_means(roberta_df)
# debiased_roberta_means = find_means(debiased_roberta_df)

# all_means = pd.concat([
#   bert_means, debiased_bert_means,
#   albert_means, debiased_albert_means,
#   roberta_means, debiased_roberta_means,
#   ], axis=1)

# all_means.columns = ["Male Score Bert", "Female Score Bert", "Male Score Bert Debiased", "Female Score Bert Debiased", 
#                      "Male Score ALBERT", "Female Score ALBERT", "Male Score ALBERT Debiased", "Female Score ALBERT Debiased",
#                      "Male Score RoBERTa", "Female Score RoBERTa", "Male Score RoBERTa Debiased", "Female Score RoBERTa Debiased"]

# print(all_means)
# all_means.plot.bar(rot=0)
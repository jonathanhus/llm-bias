import pandas as pd

'''
The NLI-Bias dataset as created using the generate templates is over over 300 MBs
in size. This can't be stored in github unless git lfs is used. Instead of using
that, this script removes some columns and rows in order to get the size under
100 MB.

'''

df = pd.read_csv('nli_bias.csv')
print(df.columns)
df = df.drop(['id', 'pair type', 'premise_filler_word', 'template_type'], axis=1)
df = df.drop(['hypothesis_filler_word'], axis=1)
print(df.columns)

# Removes everything after the sheriff occupation
df = df.iloc[:1641312]
df.to_csv('reduced_nli.csv', index=False)
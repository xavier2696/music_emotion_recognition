import pandas as pd

print('Combining feature dataframes')
df1 = pd.read_csv('data/features1.csv')
df2 = pd.read_csv('data/features2.csv')
df3 = pd.read_csv('data/features3.csv')
df4 = pd.read_csv('data/features4.csv')
combined_df = pd.concat([df1, df2, df3, df4])
combined_df.to_csv('data/features.csv', index=False)

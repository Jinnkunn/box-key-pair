import pandas as pd
import os

long_df = pd.read_csv('data/dollhouse_long.csv')
short_df = pd.read_csv('data/dollhouse_short.csv.csv')

gender_age_dict = dict(zip(short_df['ID'], zip(short_df['Gender'], short_df['Age'])))

for idx, row in long_df.iterrows():
    if row['ID'] in gender_age_dict:
        gender, age = gender_age_dict[row['ID']]
        long_df.at[idx, 'Gender'] = gender
        long_df.at[idx, 'Age'] = age

output_path = 'data/dollhouse.csv'
long_df.to_csv(output_path, index=False)

print(f"Data saved to: {output_path}")

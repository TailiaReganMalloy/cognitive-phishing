import pandas as pd 

df = pd.read_excel("Cognitive-Bias-Approach/congtive/dataset_excel/datasets_english_and_tag.xlsx")
print(df.columns)
"""
Index(['Unnamed: 0', 'text', 'Authority Bias', 'Survivorship Bias',
       'Pessimism Bias', 'Zero-Risk Bias', 'Hyperbolic Discounting',
       'Identifiable Victim Effect', 'Appeal to Novelty', 'Urgency Effect',
       'Curiosity ', 'Conformity', 'text_translated'],
      dtype='str')
"""
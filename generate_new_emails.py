import pandas as pd 
from Utilities import E
Emails_Chosen = pd.read_csv("./Data/Emails_Chosen.csv")

print(Emails_Chosen.columns)

for idx, Email in Emails_Chosen.iterrows():
    print(Email)

    assert(False)


import pandas as pd 

Emails_Formatted = pd.read_csv("Emails_Formatted.csv")
"""
Index(['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'EmailId', 'BaseEmailID',
       'Author', 'Style', 'Type', 'Sender Style', 'Sender', 'Subject',
       'Sender Mismatch', 'Request Credentials', 'Subject Suspicious',
       'Urgent', 'Offer', 'Link Mismatch', 'Prompt', 'Body',
       'LLM Raw Response', 'LLM Parse Error', 'LLM Authority bias',
       'LLM Scarcity bias', 'LLM Urgency bias', 'LLM Anchoring effect',
       'LLM Conformity bias', 'LLM Overconfidence', 'LLM Familiarity bias',
       'SelectedEmailType', 'SelectionMethod', 'New Email Body',
       'New Email Response', 'New Email Prompt', 'New Email Config',
       'New Email Model Name', 'Target Biases Used', 'Generation Status',
       'Generation Error', 'Generated At UTC', 'combined_target_score',
       'Scarcity', 'Urgency', 'Anchoring', 'Conformity', 'Overconfidence',
       'Familiarity', 'GPT Email Body'],
      dtype='str')
"""

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 796, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 796, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 844, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 844, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 846, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 846, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 1280, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 1280, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 1072, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 1072, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 96, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 96, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 204, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 204, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 668, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 668, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 720, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 720, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 368, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 368, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 406, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 406, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 555, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 555, 'GPT Email Body']
)


Emails_Formatted.loc[Emails_Formatted['EmailId'] == 330, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 330, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 436, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 436, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 396, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 396, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 56, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 56, 'GPT Email Body']
)


Emails_Formatted.loc[Emails_Formatted['EmailId'] == 4, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 4, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 551, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 551, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 664, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 664, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 702, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 702, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 336, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 336, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 200, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 200, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 740, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 740, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 656, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 656, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 276, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 276, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 660, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 660, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 732, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 732, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 640, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 640, 'GPT Email Body']
)

Emails_Formatted.loc[Emails_Formatted['EmailId'] == 428, 'New Email Body'] = (
    Emails_Formatted.loc[Emails_Formatted['EmailId'] == 428, 'GPT Email Body']
)



# Remove auto-generated index columns (Unnamed: ...), then save without index.
Emails_Formatted = Emails_Formatted.loc[
    :, ~Emails_Formatted.columns.astype(str).str.startswith("Unnamed:")
]
Emails_Formatted.to_csv("Emails_Formatted.csv", index=False)
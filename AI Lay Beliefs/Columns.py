import pandas as pd


study_1a = pd.read_spss("./Study1A_Data.sav")

"""
Index(['StartDate', 'EndDate', 'Status', 'Progress', 'Durationinseconds',
       'Finished', 'RecordedDate', 'ResponseId', 'DistributionChannel',
       'UserLanguage', 'mobilet_PageSubmit', 'Q1_Browser', 'Q1_Version',
       'Q1_OperatingSystem', 'Q1_Resolution', 'consent', 'Condition', 'dv',
       'dv_t', 'age', 'sex', 'sex_3_TEXT', 'langdiffi', 'langdiffi_2_TEXT',
       'techdiffi', 'techdiffi_2_TEXT', 'comments', 'FL_19_DO'],
      dtype='str')
"""
study_1b = pd.read_spss("./Study1B_Data.sav")

"""
Index(['StartDate', 'EndDate', 'Status', 'Progress', 'Durationinseconds',
       'Finished', 'RecordedDate', 'ResponseId', 'DistributionChannel',
       'UserLanguage', 'mobilet_PageSubmit', 'Q1_Browser', 'Q1_Version',
       'Q1_OperatingSystem', 'Q1_Resolution', 'consent', 'humreason',
       'humfeeling', 'humindex', 'humdec_t_PageSubmit', 'AIreason',
       'AIfeeling', 'aiindex', 'AIdec_t_PageSubmit', 'age', 'sex',
       'sex_3_TEXT', 'langdiffi', 'langdiffi_2_TEXT', 'techdiffi',
       'techdiffi_2_TEXT', 'comments', 'FL_19_DO', 'Human_DO', 'AI_DO',
       'Order'],
      dtype='str')
"""
study_2  = pd.read_spss("./Study2_Data.sav")
"""
Index(['StartDate', 'EndDate', 'Status', 'Progress', 'Durationinseconds',
       'Finished', 'RecordedDate', 'ResponseId', 'DistributionChannel',
       'UserLanguage', 'mobilet_PageSubmit', 'Q1_Browser', 'Q1_Version',
       'Q1_OperatingSystem', 'Q1_Resolution', 'consent', 'IV',
       'humancondition', 'AIcondition', 'huloadt_PageSubmit',
       'ailoadt_PageSubmit', 'waitingt_PageSubmit', 'decision', 'DV', 'reason',
       'feeling', 'Index', 'age', 'sex', 'sex_3_TEXT', 'langdiffi',
       'langdiffi_2_TEXT', 'techdiffi', 'techdiffi_2_TEXT', 'comments',
       'FL_19_DO', 'DecisionMakingStyle_DO', 'filter_$'],
      dtype='str')
"""
print(study_2.columns)
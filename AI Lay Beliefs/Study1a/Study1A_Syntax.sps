* Encoding: UTF-8.

DATASET ACTIVATE DataSet1.
ONEWAY dv BY Condition
  /ES=OVERALL
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

USE ALL.
COMPUTE filter_$=(Condition = 1).
VARIABLE LABELS filter_$ 'Condition = 1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.

T-TEST
  /TESTVAL=5
  /MISSING=ANALYSIS
  /VARIABLES=dv
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

USE ALL.
COMPUTE filter_$=(Condition  ~=  1).
VARIABLE LABELS filter_$ 'Condition  ~=  1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.

T-TEST
  /TESTVAL=5
  /MISSING=ANALYSIS
  /VARIABLES=dv
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).



* Encoding: UTF-8.

DATASET ACTIVATE DataSet1.
LOGISTIC REGRESSION VARIABLES DV
  /METHOD=ENTER IV 
  /CONTRAST (IV)=Indicator
  /CRITERIA=PIN(0.05) POUT(0.10) ITERATE(20) CUT(0.5).


CROSSTABS
  /TABLES=IV BY DV
  /FORMAT=AVALUE TABLES
  /STATISTICS=CHISQ 
  /CELLS=COUNT ROW 
  /COUNT ROUND CELL.

ONEWAY Index BY IV
  /ES=OVERALL
  /STATISTICS DESCRIPTIVES 
  /MISSING ANALYSIS
  /CRITERIA=CILEVEL(0.95).

/* Run Process Macro, Model 4, Specifications: X= IV, Y= DV, M= Feeling MINUS Reason [Index], 95%CI, 5000 bootstraps
    
/* For the next analysis, select only the 'Human' Experimental Condition or run the code below:
    
USE ALL.
COMPUTE filter_$=(langdiffi = 1 AND techdiffi = 1 AND humancondition = 1).
VARIABLE LABELS filter_$ 'langdiffi = 1 AND techdiffi = 1 AND humancondition = 1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.

T-TEST
  /TESTVAL=0
  /MISSING=ANALYSIS
  /VARIABLES=Index
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

/* For the next analysis, select only the 'AI' Experimental Condition or run the code  below:
    

USE ALL.
COMPUTE filter_$=(langdiffi = 1 AND techdiffi = 1 AND AIcondition = 1).
VARIABLE LABELS filter_$ 'langdiffi = 1 AND techdiffi = 1 AND AIcondition = 1 (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.

T-TEST
  /TESTVAL=0
  /MISSING=ANALYSIS
  /VARIABLES=Index
  /ES DISPLAY(TRUE)
  /CRITERIA=CI(.95).

/*To go back to filtering as per preregistration again, use the code below:
    

USE ALL.
COMPUTE filter_$=(langdiffi = 1 AND techdiffi = 1 ).
VARIABLE LABELS filter_$ 'langdiffi = 1 AND techdiffi = 1  (FILTER)'.
VALUE LABELS filter_$ 0 'Not Selected' 1 'Selected'.
FORMATS filter_$ (f1.0).
FILTER BY filter_$.
EXECUTE.


/*For the paralell mediation reposrted in the Appendix A, use the Process model as per the below specification:
    
/* Run Process Macro, Model 4, Specifications: X= IV, Y= DV, M1= feeling, M2= reason [Index], 95%CI, 5000 bootstraps

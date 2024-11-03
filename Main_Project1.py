'''
Script for ECO 2425 Project 1

I read the data files directly from the replication package
the 'pulse_w2845'. The data can be found in the replication package
here: 

Parolin, Z., Ananat, E., Collyer, S., Curran, M., & Wimer, C. (2023). 
The Effects of the Monthly and Lump-Sum Child Tax Credit Payments on Food and Housing Hardship. 
AEA Papers and Proceedings, 113, 406-412. https://doi.org/10.1257/pandp.20231088.

Earliest Date: April 14 – April 26 2021 #Week 28
Latest Date: April 27 – May 9, 2022 #Week 45

Sources:

In the "Data Clean up" Section, I translated the code from Stata to Python.
The original Stata code is in their replication package https://doi.org/10.1257/pandp.20231088.
I only do some slight modifications to include an extra variable. 

In the "Dif-in-Dif model set-up" and "WLS regressions", I use almost the exact
same model as the original paper. https://doi.org/10.1257/pandp.20231088.

In "Figures Proof of Parallel Trends" I make a similar graph to what the authors
made.

In the "Directed Acyclic Graph (DAG)" section, I mainly use the code from
Week 5 and 6 Notebooks on Quercus. I make some modifications to fit with
my project.

In the "Machine Learning Setup (excluding DAG)", "Ridge regression" and "Lasso regression"
I use code from the Week 3 notebooks on Quercus. But I have made some modifications to adapt to
a binary case.

In the "Regression Tree" and "Random Forest", I take the code from Week 4 notebooks
on Quercus.

'''


#%%% Importing packages

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from statsmodels.iolib.summary2 import summary_col
from matplotlib.pyplot import subplots
import sklearn.model_selection as skm
from ISLP.models import ModelSpec as MS
from sklearn.tree import (DecisionTreeClassifier as DTC,
                          plot_tree,
                          export_text)
from sklearn.ensemble import \
     (RandomForestClassifier as RFC)
from dowhy import CausalModel
import networkx as nx
from stargazer.stargazer import Stargazer

#%%% Preliminaries
#Change the file paths appropriate
pulse_file = 'C:/Users/Owner/Desktop/UofT Classes/ECO2425/Project/Raw/data/pulse_w2845.dta'

#Reading in the data
pulse = pd.read_stata(pulse_file)

    
#%%%Data Clean up----------------------------------------------------------

#Changing some column data types
pulse['month'] = pulse['month'].astype(int)
pulse['year'] = pulse['year'].astype(int)

pulse['est_st'] = pulse['est_st'].astype(str)
pulse['month'] = pulse['month'].astype(str)
pulse['year'] = pulse['year'].astype(str)


#From here onwards, I translate the coding from Stata to Python.
#I just translate it to Python.
pulse['snap_emergallot'] = 1

pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '9')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '9')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '10')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '10')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '11')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '11')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '12')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '12')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '1') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '2') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '3') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '4') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '1') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '2') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '4') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '5') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '6') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '8') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '9') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '10') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '11') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '12') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '13') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '15') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '16') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '17') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '18') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '19') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '20') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '21') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '22') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '23') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '24') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '25') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '26') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '27') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '28') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '29') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '30') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '31') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '32') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '33') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '34') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '35') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '36') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '37') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '38') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '39') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '40') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '41') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '42') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '44') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '45') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '46') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '47') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 0, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '48') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '49') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '50') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '51') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '53') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '54') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '55') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])
pulse['snap_emergallot'] = np.where(((pulse['est_st'] ==  '56') & (pulse['month'] ==  '5') & (pulse['year'] == '2022')), 1, pulse['snap_emergallot'])

#Creating the inverse just like the authors
pulse['lostsnap'] = np.where(pulse['snap_emergallot'] == 0,
                             1,
                             0)
########################UI multiplier ##################################3
pulse['ui_multiplier'] = 1

pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='1')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='2')), 0.4 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='5')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '7')  & (pulse['est_st'] =='4')), 0.322580645 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='12')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='13')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='16')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='18')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='19')), 0.4 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '7')  & (pulse['est_st'] =='24')), 0.096774194 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='28')), 0.4 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='29')), 0.4 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='30')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='31')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='33')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='38')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='39')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='40')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='45')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='46')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '7')  & (pulse['est_st'] =='47')), 0.096774194 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='48')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='49')), 0.866666667 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='54')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '6')  & (pulse['est_st'] =='56')), 0.633333333 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']== '8')  & (pulse['est_st'] =='22')), 0.096774194 , pulse['ui_multiplier'])

pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='1')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='2')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='5')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '7')  & (pulse['est_st'] =='4')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='12')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='13')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='16')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='18')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='19')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '7')  & (pulse['est_st'] =='24')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='28')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='29')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='30')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='31')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='33')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='38')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='39')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='40')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='45')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='46')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '7')  & (pulse['est_st'] =='47')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='48')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='49')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='54')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '6')  & (pulse['est_st'] =='56')), 0 , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where(((pulse['month']> '8')  & (pulse['est_st'] =='22')), 0 , pulse['ui_multiplier'])

pulse['ui_multiplier'] = np.where(((pulse['month']> '8')  & (pulse['year'] =='2021')), 0  , pulse['ui_multiplier'])
pulse['ui_multiplier'] = np.where((pulse['year']== '2022'), 0, pulse['ui_multiplier'])

#Calculating if state has lost ui, following the paper
pulse['lostui'] = 1 - pulse['ui_multiplier']
pulse['lostui'] = np.where(((pulse['month'].isin(['9','10','11','12'])) &
                           (pulse['year'] == '2021')) |
                           (pulse['year'] == '2022'), 
                           1,
                           pulse['lostui'])

#Create a binary version of lostui... Using authors' naming convention
pulse['lostui_01'] = pulse['lostui']
pulse['lostui_01'] = np.where(pulse['lostui'] < 0.5,
                              0,
                              1)

#Whether a state lost SNAP
pulse['lostsnap'] = np.where(pulse['snap_emergallot'] == 0,
                             1,
                             0)

#Net_EITC divide by 1000 following what the authors did
pulse['net_eitc_1000'] = pulse['net_eitc']/1000

##Create post treatment variables to indicate when treated

#Monthly Payment
pulse['post1'] = 0
pulse['post1'] = np.where(pulse['week'].isin([34,
                                              35,
                                              36,
                                              37,
                                              38,
                                              39,
                                              40]), 1, pulse['post1'])

#Lump-sum Payments
pulse['post2'] = 0
pulse['post2'] = np.where(pulse['week'].isin([43,
                                              44,
                                              45]), 1, pulse['post2'])


#Where there was Partial payment
pulse['partial'] = 0
pulse['partial'] = np.where(pulse['week'].isin([41,
                                                42]), 1, pulse['partial'])

#Pre-treatment period
pulse['pre'] = 0
pulse['pre'] = np.where(pulse['week'].isin([28,
                                            29,
                                            30,
                                            31,
                                            32,
                                            33]), 1, pulse['pre'])


#Marital status, this is where I code in Unmarried. It wasn't included originally
#Keep only the non-missing values. Anything 1 and above the survey respondent answer normally
pulse = pulse[pulse['ms'] >= 1]

#Creating a dummy for if the person is unmarried or not
#If unmarried then 1, else 0
pulse['unmarried'] = np.where(pulse['ms'].isin([2,3,4,5]),
                              1,
                              0)

#Creating interaction terms like what the authors did
pulse['ctc_treatment_01_post1'] = pulse['ctc_treatment_01']*pulse['post1']
pulse['ctc_treatment_01_post2'] = pulse['ctc_treatment_01']*pulse['post2']
pulse['ctc_treatment_01_partial'] = pulse['ctc_treatment_01']*pulse['partial']
pulse['ctc_treatment_01_lostsnap'] = pulse['ctc_treatment_01']*pulse['lostsnap']
pulse['ctc_treatment_01_lostui'] = pulse['ctc_treatment_01']*pulse['lostui']

#Controls from the base replication package (except I add unmarried).
controls = ['net_eitc_1000',
            'age',
            'age2',
            'v_female',
            'edu_college',
            'edu_nohs',
            'ctc_treatment_01_partial',
            'ctc_treatment_01_lostsnap',
            'ctc_treatment_01_lostui',
            'ctc_treatment_01',
            'post1',
            'post2',
            'partial',
            'lostsnap',
            'lostui',
            'unmarried',
            'statefips',
            'week'
            ]

#My variables of interest
other_vars = [
    'ctc_treatment_01_post2', #Parameter of interest in DD
    'ctc_treatment_01_post1',#Parameter of interest in DD
    ] 

##Keep only the variables used in the analysis
pulse_foodhardship = pulse[[
                'p_foodhardship',  #Dependent variable
                'pweight', #population weight
                'rhispanic',
                'rrace'
                ] + 
                other_vars +
                controls] #Separate by Hispanic and Non-Hispanic

#Drop nas in the dataset
pulse_foodhardship = pulse_foodhardship.dropna()

#Make these into categoricals, this will make it easier to work with later
pulse_foodhardship = pulse_foodhardship.astype({'statefips' : 'category',
                                              'week' : 'category'})

#Indicating which is Hispanic vs non-hispanic
hispanic_dict = {
    1 : "Non-Hispanic",
    2 : "Hispanic"
    }
#Race of the household head
race_dict = {
    1 : "White",
    2 : "Black",
    3 : "Asian",
    4 : "Other races"}

#Replacing the names
pulse_foodhardship['rhispanic'].replace(hispanic_dict,inplace=True)
pulse_foodhardship['rrace'].replace(race_dict,inplace=True)

#Make a new column, Hispanic if they identify as hispanic. Non-hispanic otherwise.
pulse_foodhardship['race'] = np.where(pulse_foodhardship['rhispanic'] == 'Hispanic',
                                      "Hispanic",
                                      pulse_foodhardship['rrace'])


#%%%Summary Statistics----------------------------------------------------------

#Creating a modified (mod) dataframe just for summary statistics
pulse_mod = pulse_foodhardship[[
                   'pweight',
                   'p_foodhardship',
                   'ctc_treatment_01',
                   'net_eitc_1000',
                   'age',
                   'v_female',
                   'edu_college',
                   'edu_nohs',
                   'unmarried',
                   'race'
                   ]]


pulse_mod['net_eitc'] = pulse_mod['net_eitc_1000']*1000 #Multiply by 1000 again
pulse_mod['ctc_treatment_01'] = np.where(pulse_mod['ctc_treatment_01'] == 1,
                     'Child in household',
                     'Child not in household') #Renaming ctc_treatment_1000

pulse_mod['Hispanic'] = np.where(pulse_mod['race'] == 'Hispanic',
                                 1,
                                 0)

pulse_mod['Black'] = np.where(pulse_mod['race'] == 'Black',
                                 1,
                                 0)

pulse_mod['Other'] = np.where(pulse_mod['race'].isin(['White','Asian','Other races']),
                                 1,
                                 0)

# Group by treated vs control group
summary = pulse_mod.groupby(['ctc_treatment_01']).apply(
    (lambda x:  pd.Series({'Food hardship' : np.average(x['p_foodhardship'], weights=x['pweight']),
                           'Female' : np.average(x['v_female'], weights=x['pweight']),
                           'College educated' : np.average(x['edu_college'], weights=x['pweight']),
                           'No highschool' : np.average(x['edu_nohs'], weights=x['pweight']),
                           'Unmarried' : np.average(x['unmarried'], weights=x['pweight']),
                           'Hispanic' : np.average(x['Hispanic'], weights=x['pweight']),
                           'Black' : np.average(x['Black'], weights=x['pweight']),
                           'Other races' : np.average(x['Other'], weights=x['pweight']),
                           'Age' : np.average(x['age'], weights=x['pweight']),
                           'Net EITC' : np.average(x['net_eitc'], weights=x['pweight'])
                           })))


# Calculate overall statistics
summary_overall = pd.Series({
   'Food hardship' : np.average(pulse_mod['p_foodhardship'], weights = pulse_mod['pweight']),
   'Female' : np.average(pulse_mod['v_female'], weights = pulse_mod['pweight']),
   'College educated' : np.average(pulse_mod['edu_college'], weights = pulse_mod['pweight']),
   'No highschool' : np.average(pulse_mod['edu_nohs'], weights = pulse_mod['pweight']),
   'Unmarried' : np.average(pulse_mod['unmarried'], weights = pulse_mod['pweight']),
   'Hispanic' : np.average(pulse_mod['Hispanic'], weights = pulse_mod['pweight']),
   'Black' : np.average(pulse_mod['Black'], weights = pulse_mod['pweight']),
   'Other races' : np.average(pulse_mod['Other'], weights = pulse_mod['pweight']),
   'Age' : np.average(pulse_mod['age'], weights = pulse_mod['pweight']),
   'Net EITC' : np.average(pulse_mod['net_eitc'], weights = pulse_mod['pweight'])
    }).reset_index()

#Make it into the same format as the summary dataframe
summary_overall = summary_overall.pivot_table(columns = 'index')
summary_overall = summary_overall.rename(index = {0 : 'All households'})


# Concat the two dataframes
summary_full = pd.concat([summary, summary_overall])

#Multiplying to convert to percentages and round
summary_full['Food hardship'] = (summary_full['Food hardship']*100).round(2)
summary_full['Female'] = (summary_full['Female']*100).round(2)
summary_full['College educated'] = (summary_full['College educated']*100).round(2)
summary_full['No highschool'] = (summary_full['No highschool']*100).round(2)
summary_full['Unmarried'] = (summary_full['Unmarried']*100).round(2)
summary_full['Hispanic'] = (summary_full['Hispanic']*100).round(2)
summary_full['Black'] = (summary_full['Black']*100).round(2)
summary_full['Other races'] = (summary_full['Other races']*100).round(2)
summary_full['Age'] = (summary_full['Age']).round(2)
summary_full['Net EITC'] = (summary_full['Net EITC']).round(2)


# Reset index
summary_full.reset_index(inplace=True)

# Reformat the DataFrame to have groups as rows
summary_melt = summary_full.melt(id_vars='index', 
                               value_vars=['Food hardship',
                                           'Female',
                                           'College educated',
                                           'No highschool',
                                           'Unmarried',
                                           'Hispanic',
                                           'Black',
                                           'Other races',
                                           'Age',
                                           'Net EITC'], 
                               var_name='Mean', 
                               value_name='Value')

# Pivot the table
summary_pivot = summary_melt.pivot(index='Mean', columns='index', values='Value')



# Export to LaTeX
latex_table = summary_pivot.to_latex(
    index=True,
    float_format="%.2f",
    header=['All households', 'Child in household' , 'Child not in household'],
    column_format="l c c c", 
    escape=False
)

#Print the Latex Code
print(latex_table)

#Count the number of observations
obs = len(pulse_mod)

#Count the number of observations with children
pulse_mod_children = pulse_mod[pulse_mod['ctc_treatment_01'] == 'Child in household']

obs_children = len(pulse_mod_children)

#Count the number of observations without children
obs_nochildren = obs - obs_children

print(obs, obs_children, obs_nochildren)


#%%%  Dif-in-Dif model set-up---------------------------------------------------------

#Difference-in-Difference model, same as the paper, except I add in unmarried in the controls
formula_dif_dif = ("p_foodhardship ~ ctc_treatment_01_post1 + ctc_treatment_01_post2 + "  + ' + '.join(controls))


#%%%  Regressions---------------------------------------------------------

#Separating out Hispanic
pulse_foodhardship_hispanic = pulse_foodhardship[pulse_foodhardship['race'] == 'Hispanic']

#Separating out Non-Hispanic Black
pulse_foodhardship_black = pulse_foodhardship[
    (pulse_foodhardship['race'] == 'Black') ]

#Separating out Non-hispanic White, Asian and Other
pulse_foodhardship_other = pulse_foodhardship[
    pulse_foodhardship['race'].isin(['White', 'Asian', 'Other'])]

#I will subsample to run my machine learning models that is using the whole population
#This will take a subsample of each combination of week and state of about 250 each
pulse_foodhardship_sub = pulse_foodhardship.groupby(['week','statefips']).apply(lambda x: x.sample(n=250, random_state=1000)).reset_index(drop=True)

######## OLS ############

# OLS for the entire sample
model_ols = smf.ols(formula_dif_dif, data=pulse_foodhardship)
results_ols = model_ols.fit(cov_type='HC3', cluster=pulse_foodhardship['statefips'])

# OLS for the subsample
model_ols_sub = smf.ols(formula_dif_dif, data=pulse_foodhardship_sub)
results_ols_sub = model_ols_sub.fit(cov_type='HC3', cluster=pulse_foodhardship_sub['statefips'])

# OLS for the Hispanic
model_ols_hispanic = smf.ols(formula_dif_dif, data=pulse_foodhardship_hispanic)
results_ols_hispanic = model_ols_hispanic.fit(cov_type='HC3', cluster=pulse_foodhardship_hispanic['statefips'])

# OLS for the Non-Hispanic Black
model_ols_black = smf.ols(formula_dif_dif, data=pulse_foodhardship_black)
results_ols_black = model_ols_black.fit(cov_type='HC3', cluster=pulse_foodhardship_black['statefips'])

# OLS for the other races
model_ols_other = smf.ols(formula_dif_dif, data=pulse_foodhardship_other)
results_ols_other = model_ols_other.fit(cov_type='HC3', cluster=pulse_foodhardship_other['statefips'])


# Stargazer
ols_table = Stargazer([results_ols,
                       results_ols_sub,
                       results_ols_hispanic, 
                       results_ols_black,
                       results_ols_other])
#Rename the columns
ols_table.custom_columns(['All', 
                         'All(subsample)', 
                         'Hispanic',
                         'Black',
                         'Other'], [1, 1, 1, 1, 1])

#Rename the variables of interest
ols_table.rename_covariates({'ctc_treatment_01_post1' : 'Monthly Payments',
                             'ctc_treatment_01_post2' : 'Lump-sum Payments' })

# Rename the dependent variable
ols_table.dependent_variable_name("Food hardship")

#Make to Latex
ols_latex = ols_table.render_latex()

#Print it out
print(ols_latex)


######## WLS ############

# wls for the entire sample
model_wls = smf.wls(formula_dif_dif, data=pulse_foodhardship, weights=pulse_foodhardship['pweight'])
results_wls = model_wls.fit(cov_type='HC3', cluster=pulse_foodhardship['statefips'])

# wls for the subsample
model_wls_sub = smf.wls(formula_dif_dif, data=pulse_foodhardship_sub, weights=pulse_foodhardship_sub['pweight'])
results_wls_sub = model_wls_sub.fit(cov_type='HC3', cluster=pulse_foodhardship_sub['statefips'])

# wls for the Hispanic
model_wls_hispanic = smf.wls(formula_dif_dif, data=pulse_foodhardship_hispanic, weights=pulse_foodhardship_hispanic['pweight'])
results_wls_hispanic = model_wls_hispanic.fit(cov_type='HC3', cluster=pulse_foodhardship_hispanic['statefips'])

# wls for the Non-Hispanic Black
model_wls_black = smf.wls(formula_dif_dif, data=pulse_foodhardship_black, weights=pulse_foodhardship_black['pweight'])
results_wls_black = model_wls_black.fit(cov_type='HC3', cluster=pulse_foodhardship_black['statefips'])

# wls for the other races
model_wls_other = smf.wls(formula_dif_dif, data=pulse_foodhardship_other, weights=pulse_foodhardship_other['pweight'])
results_wls_other = model_wls_other.fit(cov_type='HC3', cluster=pulse_foodhardship_other['statefips'])


# Stargazer
wls_table = Stargazer([results_wls,
                       results_wls_sub,
                       results_wls_hispanic, 
                       results_wls_black,
                       results_wls_other])

#Rename the columns
wls_table.custom_columns(['All', 
                         'All', #Subsample 
                         'Hispanic',
                         'Black',
                         'Other'], [1, 1, 1, 1, 1])

#Rename the variables of interest
wls_table.rename_covariates({'ctc_treatment_01_post1' : 'Monthly Payments',
                             'ctc_treatment_01_post2' : 'Lump-sum Payments' })

# Rename the dependent variable
wls_table.dependent_variable_name("Food hardship")

#Make to Latex
wls_latex = wls_table.render_latex()

#Print
print(wls_latex)


#%%% Figures Proof of Parallel Trends---------------------------------------------------------

####Figure for the parallel trend assumption for households with children vs no children

###### All races and ethnicities ######
#First keep only whatever variables are necessary
pulse_foodhardship_fig1 = pulse_foodhardship[['p_foodhardship','pweight','ctc_treatment_01','week']]

##Dataframe for parents (Households with children)
pulse_foodhardship_fig1_parents = pulse_foodhardship_fig1[
    pulse_foodhardship_fig1['ctc_treatment_01'] == 1].reset_index(drop = True)

#Calculate the mean of each period for parents
pulse_foodhardship_fig1_parents_mean = pulse_foodhardship_fig1_parents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))

#Convert to a dataframe
pulse_foodhardship_fig1_parents_mean = pd.DataFrame(
    pulse_foodhardship_fig1_parents_mean).reset_index()

#Rename column
pulse_foodhardship_fig1_parents_mean = pulse_foodhardship_fig1_parents_mean.rename(
    columns = {0 : 'Children present in household'})

#Dataframe for non-parents (Households without Children)
pulse_foodhardship_fig1_noparents = pulse_foodhardship_fig1[
    pulse_foodhardship_fig1['ctc_treatment_01'] == 0].reset_index(drop = True)

#Calculate the mean of each period for non-parents
pulse_foodhardship_fig1_noparents_mean = pulse_foodhardship_fig1_noparents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))


#Convert to a dataframe
pulse_foodhardship_fig1_noparents_mean = pd.DataFrame(
   pulse_foodhardship_fig1_noparents_mean).reset_index()

#Rename column
pulse_foodhardship_fig1_noparents_mean = pulse_foodhardship_fig1_noparents_mean.rename(
    columns = {0 : 'Children not present in household'})

#Merge the two to create one dataframe
figure1 = pd.merge(pulse_foodhardship_fig1_parents_mean, pulse_foodhardship_fig1_noparents_mean)


#First keep only whatever variables are necessary
pulse_foodhardship_fig2 = pulse_foodhardship_sub[['p_foodhardship','pweight','ctc_treatment_01','week']]

##Dataframe for parents
pulse_foodhardship_fig2_parents = pulse_foodhardship_fig2[
    pulse_foodhardship_fig2['ctc_treatment_01'] == 1].reset_index(drop = True)

#Calculate the mean of each period for parents
pulse_foodhardship_fig2_parents_mean = pulse_foodhardship_fig2_parents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))

#Convert to a dataframe
pulse_foodhardship_fig2_parents_mean = pd.DataFrame(
    pulse_foodhardship_fig2_parents_mean).reset_index()

#Rename column
pulse_foodhardship_fig2_parents_mean = pulse_foodhardship_fig2_parents_mean.rename(
    columns = {0 : 'Children present in household'})

#Dataframe for non-parents
pulse_foodhardship_fig2_noparents = pulse_foodhardship_fig2[
    pulse_foodhardship_fig2['ctc_treatment_01'] == 0].reset_index(drop = True)

#Calculate the mean of each period for non-parents
pulse_foodhardship_fig2_noparents_mean = pulse_foodhardship_fig2_noparents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))


#Convert to a dataframe
pulse_foodhardship_fig2_noparents_mean = pd.DataFrame(
   pulse_foodhardship_fig2_noparents_mean).reset_index()

#Rename column
pulse_foodhardship_fig2_noparents_mean = pulse_foodhardship_fig2_noparents_mean.rename(
    columns = {0 : 'Children not present in household'})

#Merge the two to create one dataframe
figure2 = pd.merge(pulse_foodhardship_fig2_parents_mean, pulse_foodhardship_fig2_noparents_mean)

######## Hispanic ########
#First keep only whatever variables are necessary
pulse_foodhardship_fig3 = pulse_foodhardship_hispanic[['p_foodhardship','pweight','ctc_treatment_01','week']]

##Dataframe for parents (Households with children)
pulse_foodhardship_fig3_parents = pulse_foodhardship_fig3[
    pulse_foodhardship_fig3['ctc_treatment_01'] == 1].reset_index(drop = True)

#Calculate the mean of each period for parents
pulse_foodhardship_fig3_parents_mean = pulse_foodhardship_fig3_parents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))

#Convert to a dataframe
pulse_foodhardship_fig3_parents_mean = pd.DataFrame(
    pulse_foodhardship_fig3_parents_mean).reset_index()

#Rename column
pulse_foodhardship_fig3_parents_mean = pulse_foodhardship_fig3_parents_mean.rename(
    columns = {0 : 'Children present in household'})

#Dataframe for non-parents (Households without Children)
pulse_foodhardship_fig3_noparents = pulse_foodhardship_fig3[
    pulse_foodhardship_fig3['ctc_treatment_01'] == 0].reset_index(drop = True)

#Calculate the mean of each period for non-parents
pulse_foodhardship_fig3_noparents_mean = pulse_foodhardship_fig3_noparents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))


#Convert to a dataframe
pulse_foodhardship_fig3_noparents_mean = pd.DataFrame(
   pulse_foodhardship_fig3_noparents_mean).reset_index()

#Rename column
pulse_foodhardship_fig3_noparents_mean = pulse_foodhardship_fig3_noparents_mean.rename(
    columns = {0 : 'Children not present in household'})

#Merge the two to create one dataframe
figure3 = pd.merge(pulse_foodhardship_fig3_parents_mean, pulse_foodhardship_fig3_noparents_mean)

######## Black ########

#First keep only whatever variables are necessary
pulse_foodhardship_fig4 = pulse_foodhardship_black[['p_foodhardship','pweight','ctc_treatment_01','week']]

##Dataframe for parents (Households with children)
pulse_foodhardship_fig4_parents = pulse_foodhardship_fig4[
    pulse_foodhardship_fig4['ctc_treatment_01'] == 1].reset_index(drop = True)

#Calculate the mean of each period for parents
pulse_foodhardship_fig4_parents_mean = pulse_foodhardship_fig4_parents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))

#Convert to a dataframe
pulse_foodhardship_fig4_parents_mean = pd.DataFrame(
    pulse_foodhardship_fig4_parents_mean).reset_index()

#Rename column
pulse_foodhardship_fig4_parents_mean = pulse_foodhardship_fig4_parents_mean.rename(
    columns = {0 : 'Children present in household'})

#Dataframe for non-parents (Households without Children)
pulse_foodhardship_fig4_noparents = pulse_foodhardship_fig4[
    pulse_foodhardship_fig4['ctc_treatment_01'] == 0].reset_index(drop = True)

#Calculate the mean of each period for non-parents
pulse_foodhardship_fig4_noparents_mean = pulse_foodhardship_fig4_noparents.groupby(['week']).apply(
    lambda x: np.average(x['p_foodhardship'], weights=x['pweight']))


#Convert to a dataframe
pulse_foodhardship_fig4_noparents_mean = pd.DataFrame(
   pulse_foodhardship_fig4_noparents_mean).reset_index()

#Rename column
pulse_foodhardship_fig4_noparents_mean = pulse_foodhardship_fig4_noparents_mean.rename(
    columns = {0 : 'Children not present in household'})

#Merge the two to create one dataframe
figure4 = pd.merge(pulse_foodhardship_fig4_parents_mean, pulse_foodhardship_fig4_noparents_mean)



#Now I want to map the weeks to the dates of the surveys
#This list here is for reference.
'''
The "week" of the pulse survey corresponds to: 
'28': 'April 14 - April 26',
'29': 'April 28 - May 10',
'30': 'May 12 - May 24',
'31': 'May 26 - June 7',
'32': 'June 9 - June 21',
'33': 'June 23 - July 5',
'34': 'July 21 - August 2',
'35': 'August 4 - August 16',
'36': 'August 18 - August 30',
'37': 'September 1 - September 13',
'38': 'September 15 - September 27',
'39': 'September 29 - October 11',
'40': 'December 1- December 13',
'41': 'December 29 - January 10',
'42': 'January 26 - February 7',
'43': 'March 2 - March 14',
'44': 'March 30 - April 11',
'45': 'April 27 - May 9',

'''

#I take the mid-way point of the dates
week_dict = {'28': '2021-04-20',
             '29': '2021-05-04',
             '30': '2021-05-18',
             '31': '2021-06-01',
             '32': '2021-06-15',
             '33': '2021-06-29',
             '34': '2021-07-27',
             '35': '2021-08-10',
             '36': '2021-08-24',
             '37': '2021-09-07',
             '38': '2021-09-21',
             '39': '2021-10-05',
             '40': '2021-12-07',
             '41': '2022-01-04',
             '42': '2022-02-01',
             '43': '2022-03-08',
             '44': '2022-04-05',
             '45': '2022-05-03',
             }
#Create a new column with the midway week points

#Figure 1
figure1['mid_week'] = figure1['week'].astype(str)
figure1['mid_week'].replace(week_dict,inplace=True)
figure1['mid_week'] = pd.to_datetime(figure1['mid_week'], format = "%Y-%m-%d")

#Figure 2
figure2['mid_week'] = figure2['week'].astype(str)
figure2['mid_week'].replace(week_dict,inplace=True)
figure2['mid_week'] = pd.to_datetime(figure2['mid_week'], format = "%Y-%m-%d")

#Figure 3
figure3['mid_week'] = figure3['week'].astype(str)
figure3['mid_week'].replace(week_dict,inplace=True)
figure3['mid_week'] = pd.to_datetime(figure3['mid_week'], format = "%Y-%m-%d")

#Figure 4
figure4['mid_week'] = figure4['week'].astype(str)
figure4['mid_week'].replace(week_dict,inplace=True)
figure4['mid_week'] = pd.to_datetime(figure4['mid_week'], format = "%Y-%m-%d")

#Create the daily date range... This is to help formatting in excel
figure_dates = pd.DataFrame(pd.date_range(figure1['mid_week'].min(),figure1['mid_week'].max(),freq='d'))

#Rename column
figure_dates = figure_dates.rename(columns = {0 : 'Date'})

#The authors defined monthly payments from July 21 - Dec 13
#And lump-sum payments from March 2 - April 22
figure_dates['Treatment_dates'] = 0

#Monthly CTC start
figure_dates['Treatment_dates'] = np.where(figure_dates['Date'] == '2021-07-21',
                                           1000,
                                           figure_dates['Treatment_dates']
                                           )

#Monthly CTC end
figure_dates['Treatment_dates'] = np.where(figure_dates['Date'] == '2021-12-13',
                                           1000,
                                           figure_dates['Treatment_dates']
                                           )

#Lumpsum CTC start
figure_dates['Treatment_dates'] = np.where(figure_dates['Date'] == '2022-03-02',
                                           1000,
                                           figure_dates['Treatment_dates']
                                           )

#Lumpsum CTC end
figure_dates['Treatment_dates'] = np.where(figure_dates['Date'] == '2022-04-22',
                                           1000,
                                           figure_dates['Treatment_dates']
                                           )

#Export to Excel
#Define appropriate file path
file = "C:/Users/Owner/Desktop/UofT Classes/ECO2425/Project/Work/Results/"

#Write to excel files
with pd.ExcelWriter(file + "Parallel_trend_charts.xlsx") as writer:
    figure1.to_excel(writer, sheet_name="All", index=False)
    figure2.to_excel(writer, sheet_name="Subsample", index=False)
    figure3.to_excel(writer, sheet_name="Hispanic", index=False)
    figure4.to_excel(writer, sheet_name="Black", index=False)
    figure_dates.to_excel(writer, sheet_name="Daily_dates", index=False)

#Once I export the data to Excel I make the charts in Excel.

#%%%Directed Acyclic Graph (DAG)'----------------------------------------------------------

#Simple DAG to illustrate the causal relationships
gml_graph = """graph [
directed 1

node [
    id p_foodhardship
    label "Food hardship"
    ]

node [
    id ctc_treatment_01_post1
    label "Monthly Payments to HH with Children"
    ]

node [
    id ctc_treatment_01_post2
    label "Lump-sum Payments to HH with Children"
    ]

node [
    id Controls
    label Controls
    ]

edge [
    source Controls
    target p_foodhardship
    ]

edge [
    source Controls
    target ctc_treatment_01_post1
    ]

edge [
    source Controls
    target ctc_treatment_01_post2
    ]

edge [
    source ctc_treatment_01_post1
    target p_foodhardship
    ]

edge [
    source ctc_treatment_01_post2
    target p_foodhardship
    ]
]
    
    """

# Get the graph
graph = nx.parse_gml(gml_graph)


#Changing parameters for the plot
layout = nx.planar_layout(graph)

#Adjust the figure size
plt.tight_layout()
fig, ax = plt.subplots(figsize=(24,10))

# Plot
nx.draw(
    G=graph, 
    with_labels=True,
    node_size=2500,
    font_color='black',
    pos = layout
)

#Now to do the DAG for the estimation
gml_graph_full = """graph [
directed 1

node [
    id p_foodhardship
    label "p_foodhardship"
    ]

node [
    id ctc_treatment_01_post1
    label "ctc_treatment_01_post1"
    ]

node [
    id ctc_treatment_01_post2
    label "ctc_treatment_01_post2"
    ]

node [
    id unmarried
    label "unmarried"
    ]

node [
    id net_eitc_1000
    label "net_eitc_1000"
    ]

node [
    id age
    label "age"
    ]

node [
    id age2
    label "age2"
    ]

node [
    id v_female
    label "v_female"
    ]

node [
    id edu_college
    label "edu_college"
    ]

node [
    id edu_nohs
    label "edu_nohs"
    ]

node [
    id ctc_treatment_01_partial
    label "ctc_treatment_01_partial"
    ]

node [
    id ctc_treatment_01_lostsnap
    label "ctc_treatment_01_lostsnap"
    ]

node [
    id ctc_treatment_01_lostui
    label "ctc_treatment_01_lostui"
    ]

node [
    id ctc_treatment_01
    label "ctc_treatment_01"
    ]

node [
    id post1
    label "post1"
    ]

node [
    id post2
    label "post2"
    ]

node [
    id partial
    label "partial"
    ]

node [
    id lostsnap
    label "lostsnap"
    ]

node [
    id lostui
    label "lostui"
    ]

node [
    id statefips
    label "statefips"
    ]

node [
    id week
    label "week"
    ]

edge [
    source ctc_treatment_01_post1
    target p_foodhardship
    ]

edge [
    source ctc_treatment_01_post2
    target p_foodhardship
    ]

edge [
    source unmarried
    target p_foodhardship
    ]

edge [
    source net_eitc_1000
    target p_foodhardship
    ]

edge [
    source age
    target p_foodhardship
    ]

edge [
    source age2
    target p_foodhardship
    ]

edge [
    source v_female
    target p_foodhardship
    ]

edge [
    source edu_college
    target p_foodhardship
    ]

edge [
    source edu_nohs
    target p_foodhardship
    ]

edge [
    source ctc_treatment_01_partial
    target p_foodhardship
    ]

edge [
    source ctc_treatment_01_lostsnap
    target p_foodhardship
    ]

edge [
    source ctc_treatment_01_lostui
    target p_foodhardship
    ]

edge [
    source ctc_treatment_01
    target p_foodhardship
    ]

edge [
    source post1
    target p_foodhardship
    ]

edge [
    source post2
    target p_foodhardship
    ]

edge [
    source partial
    target p_foodhardship
    ]

edge [
    source lostsnap
    target p_foodhardship
    ]

edge [
    source lostui
    target p_foodhardship
    ]

edge [
    source statefips
    target p_foodhardship
    ]

edge [
    source week
    target p_foodhardship
    ]

edge [
    source unmarried
    target ctc_treatment_01_post1
    ]

edge [
    source net_eitc_1000
    target ctc_treatment_01_post1
    ]

edge [
    source age
    target ctc_treatment_01_post1
    ]

edge [
    source age2
    target ctc_treatment_01_post1
    ]

edge [
    source v_female
    target ctc_treatment_01_post1
    ]

edge [
    source edu_college
    target ctc_treatment_01_post1
    ]

edge [
    source edu_nohs
    target ctc_treatment_01_post1
    ]

edge [
    source ctc_treatment_01_partial
    target ctc_treatment_01_post1
    ]

edge [
    source ctc_treatment_01_lostsnap
    target ctc_treatment_01_post1
    ]

edge [
    source ctc_treatment_01_lostui
    target ctc_treatment_01_post1
    ]

edge [
    source ctc_treatment_01
    target ctc_treatment_01_post1
    ]

edge [
    source post1
    target ctc_treatment_01_post1
    ]

edge [
    source post2
    target ctc_treatment_01_post1
    ]

edge [
    source partial
    target ctc_treatment_01_post1
    ]

edge [
    source lostsnap
    target ctc_treatment_01_post1
    ]

edge [
    source lostui
    target ctc_treatment_01_post1
    ]

edge [
    source statefips
    target ctc_treatment_01_post1
    ]

edge [
    source week
    target ctc_treatment_01_post1
    ]

edge [
    source unmarried
    target ctc_treatment_01_post2
    ]

edge [
    source net_eitc_1000
    target ctc_treatment_01_post2
    ]

edge [
    source age
    target ctc_treatment_01_post2
    ]

edge [
    source age2
    target ctc_treatment_01_post2
    ]

edge [
    source v_female
    target ctc_treatment_01_post2
    ]

edge [
    source edu_college
    target ctc_treatment_01_post2
    ]

edge [
    source edu_nohs
    target ctc_treatment_01_post2
    ]

edge [
    source ctc_treatment_01_partial
    target ctc_treatment_01_post2
    ]

edge [
    source ctc_treatment_01_lostsnap
    target ctc_treatment_01_post2
    ]

edge [
    source ctc_treatment_01_lostui
    target ctc_treatment_01_post2
    ]

edge [
    source ctc_treatment_01
    target ctc_treatment_01_post2
    ]

edge [
    source post1
    target ctc_treatment_01_post2
    ]

edge [
    source post2
    target ctc_treatment_01_post2
    ]

edge [
    source partial
    target ctc_treatment_01_post2
    ]

edge [
    source lostsnap
    target ctc_treatment_01_post2
    ]

edge [
    source lostui
    target ctc_treatment_01_post2
    ]

edge [
    source statefips
    target ctc_treatment_01_post2
    ]

edge [
    source week
    target ctc_treatment_01_post2
    ]
]
    
    """



#Use the subsample of the data to perform the Causal Model Estimation

######Monthly payments

model = CausalModel(
    data=pulse_foodhardship_sub,
    treatment='ctc_treatment_01_post1', 
    outcome= 'p_foodhardship', 
    graph = gml_graph_full
    )

#Identifying the relationships 
estimand = model.identify_effect()


#Doing the estimate
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name="backdoor.linear_regression")

#Refuting
refute_subset = model.refute_estimate(
estimand=estimand,
estimate=estimate,
method_name="data_subset_refuter",
subset_fraction=0.5,
random_seed = 1000)



##### Now for the Lumpsum payments
model2 = CausalModel(
    data=pulse_foodhardship_sub,
    treatment='ctc_treatment_01_post2', 
    outcome= 'p_foodhardship', 
    graph = gml_graph_full
    )

#Identifying the relationships 
estimand2 = model2.identify_effect()


#Doing the estimate
estimate2 = model2.estimate_effect(
    identified_estimand=estimand2,
    method_name="backdoor.linear_regression")

#Refute
refute_subset2 = model.refute_estimate(
estimand=estimand2,
estimate=estimate2,
method_name="data_subset_refuter",
subset_fraction=0.5,
random_seed = 1000)

#Hispanic

######Monthly payments

model_hispanic_monthly = CausalModel(
    data=pulse_foodhardship_hispanic,
    treatment='ctc_treatment_01_post1', 
    outcome= 'p_foodhardship', 
    graph = gml_graph_full
    )

#Identifying the relationships 
estimand_hispanic_monthly = model_hispanic_monthly.identify_effect()


#Doing the estimate
estimate_hispanic_monthly = model_hispanic_monthly.estimate_effect(
    identified_estimand=estimand_hispanic_monthly,
    method_name="backdoor.linear_regression")

#Refuting
refute_subset_hispanic_monthly = model_hispanic_monthly.refute_estimate(
estimand=estimand_hispanic_monthly,
estimate=estimate_hispanic_monthly,
method_name="data_subset_refuter",
subset_fraction=0.5,
random_seed = 1000)



##### Now for the Lumpsum payments
model_hispanic_lumpsum = CausalModel(
    data=pulse_foodhardship_hispanic,
    treatment='ctc_treatment_01_post2', 
    outcome= 'p_foodhardship', 
    graph = gml_graph_full
    )

#Identifying the relationships 
estimand_hispanic_lumpsum = model_hispanic_lumpsum.identify_effect()


#Doing the estimate
estimate_hispanic_lumpsum = model_hispanic_lumpsum.estimate_effect(
    identified_estimand=estimand_hispanic_lumpsum,
    method_name="backdoor.linear_regression")

#Refute
refute_subset_hispanic_lumpsum = model_hispanic_lumpsum.refute_estimate(
estimand=estimand_hispanic_lumpsum,
estimate=estimate_hispanic_lumpsum,
method_name="data_subset_refuter",
subset_fraction=0.5,
random_seed = 1000)

#Black

######Monthly payments

model_black_monthly = CausalModel(
    data=pulse_foodhardship_black,
    treatment='ctc_treatment_01_post1', 
    outcome= 'p_foodhardship', 
    graph = gml_graph_full
    )

#Identifying the relationships 
estimand_black_monthly = model_black_monthly.identify_effect()


#Doing the estimate
estimate_black_monthly = model_black_monthly.estimate_effect(
    identified_estimand=estimand_black_monthly,
    method_name="backdoor.linear_regression")

#Refuting
refute_subset_black_monthly = model_black_monthly.refute_estimate(
estimand=estimand_black_monthly,
estimate=estimate_black_monthly,
method_name="data_subset_refuter",
subset_fraction=0.5,
random_seed = 1000)



##### Now for the Lumpsum payments
model_black_lumpsum = CausalModel(
    data=pulse_foodhardship_black,
    treatment='ctc_treatment_01_post2', 
    outcome= 'p_foodhardship', 
    graph = gml_graph_full
    )

#Identifying the relationships 
estimand_black_lumpsum = model_black_lumpsum.identify_effect()


#Doing the estimate
estimate_black_lumpsum = model_black_lumpsum.estimate_effect(
    identified_estimand=estimand_black_lumpsum,
    method_name="backdoor.linear_regression")

#Refute
refute_subset_black_lumpsum = model_black_lumpsum.refute_estimate(
estimand=estimand_black_lumpsum,
estimate=estimate_black_lumpsum,
method_name="data_subset_refuter",
subset_fraction=0.5,
random_seed = 1000)




#Print all the results
# All
print(refute_subset)
print(refute_subset2)
# Hispanic
print(refute_subset_hispanic_monthly)
print(refute_subset_hispanic_lumpsum)

# Black
print(refute_subset_black_monthly)
print(refute_subset_black_lumpsum)



#%%%Machine Learning Setup (excluding DAG)----------------------------------------------------------

#Create a new dataframe from this section onwards.
ml_data = pulse_foodhardship_sub

#I want to do it for all races and ethnicities so drop the unecessary columns
ml_data = ml_data.drop(columns=['rhispanic','rrace','race'], axis=1)

ml_data = ml_data.rename(columns = {
    "p_foodhardship" : "Food Hardship",
    "ctc_treatment_01_post2" : "Lump-sum Payments to Households with Children",
    "ctc_treatment_01_post1" : "Monthly Payments to Households with Children",
    "unmarried" : "Unmarried",
    "net_eitc_1000" : "Net EITC",
    "age" : "Age",
    "age2" : "Age^2",
    "v_female" : "Female",
    "edu_college" : "College Education",
    "edu_nohs" : "No Highschool Graduation",
    "ctc_treatment_01_partial" : "Partial Payments to Households with Children",
    "ctc_treatment_01_lostsnap" : "Lost SNAP for Households with Children",
    "ctc_treatment_01_lostui" : "Lost FPUC for Households with Children",
    "ctc_treatment_01" : "Households with Children",
    "post1" : "Monthly Payments",
    "post2" : "Lump-sum Payments",
    "partial" : "Partial Payments",
    "lostsnap" : "Lost SNAP",
    "lostui" : "Lost FPUC",
    "statefips" : "States",
    "week" : "Week"})


#The full dataset is to cumbersome for my device to run so I will use the subsample
model = MS(ml_data.columns.drop(['Food Hardship','pweight']), intercept=False)

#Use code from class
D = model.fit_transform(ml_data)
feature_names = list(D.columns)
X = np.asarray(D)

Y = np.array(ml_data['Food Hardship'])

#Want to extract the weights by itself... Useful when this is split
#Into training and testing.
weight = np.asarray(ml_data['pweight'])

##Split data into training and test set. 50/50.
(X_train,
 X_test,
 Y_train,
 Y_test,
 weight_train,
 weight_test) = skm.train_test_split(X,
                                Y,
                                weight,
                                test_size=0.5,
                                random_state=0)
                                
                                

#%%%Regression Tree----------------------------------------------------------

#Define the classifier tree, code like in class
clf = DTC(criterion='entropy',
          max_depth=2,
          random_state=10) 
       
#Fit using the subsample
clf.fit(X, Y, sample_weight=weight)

ax = subplots(figsize=(50,25))[1]
plot_tree(clf,
          feature_names=feature_names,
          ax=ax,
          fontsize=30);

print(export_text(clf,
                  feature_names=feature_names,
                  show_weights=True))

#%%%Random Forest----------------------------------------------------------

#Using the subsample

#Including the state fixed effects and time fixed effects, code like in class
foodhardship_RFC_fit_full = RFC(max_features = "sqrt",
                           random_state = 10).fit(X,Y, sample_weight = weight) 

#Importance table.
feature_imp = pd.DataFrame(
    {'importance':foodhardship_RFC_fit_full.feature_importances_},
    index=feature_names)

feature_imp.sort_values(by='importance', ascending=False)


#%%%Ridge regression----------------------------------------------------------

#Take the outcome variable out and population weight out
design = MS(ml_data.columns.drop(['Food Hardship','pweight'])).fit(ml_data)


#Setup, just like the code in class
D = design.fit_transform(ml_data)
D = D.drop('intercept', axis=1)
X = np.asarray(D)

#Standardize
Xs = X - X.mean(0)[None,:]
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
#Create an array of 100 values between 10^8 to 10^-2
lambdas = 10**np.linspace(8, -2, 100) / Y.std()



# Logistic regressions has a regularization strength being the inverse of C
C_values = 1 / lambdas

# Creating a dataframe of zeroes. Rows are the amount of C values (or)
#Equivalently how many lambda values and the amount of columns from Xs
soln_array = np.zeros((len(C_values), Xs.shape[1]))

          
# Define the model, do the fit, and put it in the 0 array
for i, C in enumerate(C_values):
    #The model a logistic ridge regression, for a certain C value (which is the inverse of lambda)
    model = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000)
    #Fit the model as usual
    model.fit(Xs, Y, sample_weight = ml_data['pweight'])
    #Put all the coefficients into each row
    soln_array[i, :] = model.coef_  

#soln_array is the array of coefficients (each row gives the coefficients of
#a model at a certain C value. Combine it with the D column names (predictors)
soln_path = pd.DataFrame(soln_array,
                         columns=D.columns,
                         index=-np.log(C_values)) 

#Name the index 
soln_path.index.name = 'negative log(C)'
soln_path

#Keep only the main variables of interest
soln_path_ridge_main = soln_path[['Monthly Payments to Households with Children',
                            'Lump-sum Payments to Households with Children',
                            'Households with Children']]

path_fig, ax = subplots(figsize=(8,8))
soln_path_ridge_main.plot(ax=ax, legend=False)
ax.set_xlabel('$-log(1/\\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left');

#Include all the variables except state and time fixed effects
soln_path_ridge_appendix = soln_path[[
  "Monthly Payments to Households with Children",
  'Lump-sum Payments to Households with Children',
  'Households with Children',
  "Unmarried",
  "Net EITC",
  "Age",
  "Age^2",
  "Female",
  "College Education",
  "No Highschool Graduation",
  "Partial Payments to Households with Children",
  "Lost SNAP for Households with Children",
  "Lost FPUC for Households with Children",
  "Households with Children",
  "Monthly Payments",
  "Lump-sum Payments",
  "Partial Payments",
  "Lost SNAP",
  "Lost FPUC"]]

path_fig, ax = subplots(figsize=(8,8))
soln_path_ridge_appendix.plot(ax=ax, legend=False)
ax.set_xlabel('$-log(1/\\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left');


#Cross-Validate (similar to code in class, but with modifications for binary)

#Creating a dictionary
param_grid = {'ridge__C': 1 / lambdas}

#5-k Fold
K = 5

kfold = skm.KFold(K,
                  random_state=0,
                  shuffle=True)

# Create the logistic regression model with cross-validation
logistic_cv_ridge = LogisticRegressionCV(Cs=C_values, #C values are 1/lambda
                                    penalty='l2',
                                    cv=kfold,
                                    scoring='neg_log_loss',  # Use log loss for binary classification
                                    max_iter=1000)

# Fit the model to the data
logistic_cv_ridge.fit(Xs, Y, sample_weight = ml_data['pweight'])

# Access the best C value
best_C = logistic_cv_ridge.C_[0]  # Best regularization strength
print("Best C value:", best_C)

# Extract log loss scores for all C values
log_loss_scores = -logistic_cv_ridge.scores_[1].mean(axis=0)  # Mean log loss
std_log_loss_scores = np.std(-logistic_cv_ridge.scores_[1], axis=0)  # Std deviation


#Plotting
ridgeCV_fig, ax = subplots(figsize=(8,8))
ax.errorbar(-np.log(1/lambdas),  # Use lambda for x axis cuz want to follow convention
            log_loss_scores,   # Mean log loss for the y-axis
            yerr=np.array(std_log_loss_scores)/ np.sqrt(K),
            fmt='o', 
            label='Log Loss')
ax.axvline(-np.log(best_C), c='k', ls='--', label='Best C')
ax.set_xlabel('$-log(1/\\lambda)$', fontsize=20)
ax.set_ylabel('Cross-validated MSE', fontsize=20);

#%%%Lasso regression----------------------------------------------------------
# I cannot run Lasso with 229,500 observations or at least it takes a long time
#to run, so I cut it down to something more manageable like 18,360

lasso_sub = ml_data.groupby(['Week','States']).apply(lambda x: x.sample(n=20, random_state=1000)).reset_index(drop=True)

#Since using a subsample of a subsample need to redefine Y
Y = np.array(lasso_sub['Food Hardship'])

#Drop food hardship, population weights and the states
design = MS(lasso_sub.columns.drop(['Food Hardship','pweight', 'States'])).fit(ml_data)


#As like Ridge, clean up using class code
D = design.fit_transform(lasso_sub)
D = D.drop('intercept', axis=1)
X = np.asarray(D)

#Standardize
Xs = X - X.mean(0)[None,:]
X_scale = X.std(0)
Xs = Xs / X_scale[None,:]
#Create an array of 100 values between 10^8 to 10^-2
lambdas = 10**np.linspace(8, -2, 100) / Y.std()

# Logistic regressions will require a C which is 1/lambda according to the documentation
C_values = 1 / lambdas

# Creating a dataframe of zeroes. Rows are the amount of C values (or)
#Equivalently how many lambda values and the amount of columns from Xs
soln_array = np.zeros((len(C_values), Xs.shape[1]))

for i, C in enumerate(C_values):
    #The model a logistic lasso, for a certain C value (which is the inverse of lambda)
    model = LogisticRegression(C=C, penalty='l1', solver='saga', max_iter=1000)
    #Fit the model as usual
    model.fit(Xs, Y, sample_weight = lasso_sub['pweight'])
    #Put all the coefficients into each row
    soln_array[i, :] = model.coef_  

soln_path = pd.DataFrame(soln_array,
                         columns=D.columns,
                         index=-np.log(C_values)) 

#Name the index 
soln_path.index.name = 'negative log(C)'
soln_path


#Keep only the main variables of interest
soln_path_lasso_main = soln_path[['Monthly Payments to Households with Children',
                            'Lump-sum Payments to Households with Children',
                            'Households with Children']]

#Plot
path_fig, ax = subplots(figsize=(8,8))
soln_path_lasso_main.plot(ax=ax, legend=False)
ax.set_xlabel('$-log(1/\\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left');

#Include all the variables except state and time fixed effects
soln_path_lasso_all = soln_path[[
  "Monthly Payments to Households with Children",
  'Lump-sum Payments to Households with Children',
  'Households with Children',
  "Unmarried",
  "Net EITC",
  "Age",
  "Age^2",
  "Female",
  "College Education",
  "No Highschool Graduation",
  "Partial Payments to Households with Children",
  "Lost SNAP for Households with Children",
  "Lost FPUC for Households with Children",
  "Monthly Payments",
  "Lump-sum Payments",
  "Partial Payments",
  "Lost SNAP",
  "Lost FPUC"]]

#Plot
path_fig, ax = subplots(figsize=(8,8))
soln_path_lasso_all .plot(ax=ax, legend=False)
ax.set_xlabel('$-log(1/\\lambda)$', fontsize=20)
ax.set_ylabel('Standardized coefficients', fontsize=20)
ax.legend(loc='upper left');


aimport numpy as np
import pandas as pd

from sklearn import svm
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#show graphics in below on browser
%matplotlib inline

job_decision = pd.read_excel("C:\Personal\ML\HR Analytics Data.xlsx", sheet_name='Data')

#exploratory analysis
job_decision.info()

#heatmap of different columns
sns.heatmap(job_decision.corr())

#finding number of null values in columns
job_decision.isnull().sum()

#scatter plot data
sns.lmplot(x= 'Percent difference CTC',y= 'Duration to accept offer', data = job_decision,
    hue = 'Status', palette = 'Set1',  fit_reg=False, scatter_kws={"s":70});

#calculating number of nan values in columns
job_decision['Duration to accept offer'].isna().sum()
job_decision['Percent difference CTC'].isna().sum()

#replacing missing values with mode
duration_mode = job_decision['Duration to accept offer'].fillna(job_decision['Duration to accept offer'].mode()[0],inplace = True)
duration_mode = job_decision['Percent difference CTC'].fillna(job_decision['Percent difference CTC'].mode()[0],inplace = True)
job_decision['Percent difference CTC'].isna().sum()

#converting Status column from categorical to binary 
type_label = np.where(job_decision['Status']=='Joined',1,0)
print(type_label)
#getting column labels
jd_features = job_decision.columns.values[1:].tolist()
factors = job_decision[['Duration to accept offer','Percent difference CTC']].values
print(factors)

#converting categorical into binary values
job_decision['Joining Bonus'] = job_decision['Joining Bonus'].replace({'NA': np.nan, 'inf': np.nan}).map({'Yes': 1, 'No': 0}).astype(float).fillna(0).astype(int)
job_decision['DOJ Extended'] = job_decision['DOJ Extended'].replace({'NA': np.nan, 'inf': np.nan}).map({'Yes': 1, 'No': 0}).astype(float).fillna(0).astype(int)
job_decision['Gender'] = job_decision['Gender'].replace({'NA': np.nan, 'inf': np.nan}).map({'Male': 1, 'Female': 0}).astype(float).fillna(0).astype(int)
job_decision['Candidate relocate actual'] = job_decision['Candidate relocate actual'].replace({'NA': np.nan, 'inf': np.nan}).map({'Yes': 1, 'No': 0}).astype(float).fillna(0).astype(int)

band_mapping = {'E0':0, 'E1': 1, 'E2':2, 'E3':3, 'E4':4, 'E5':5, 'E6':6}
job_decision['Offered band'] = job_decision['Offered band'].replace({'NA':np.nan, 'inf':np.nan}).map(band_mapping).astype(float).fillna(0).astype(int)

#job_decision.head()

#defining X and Y variables for modeling
X = job_decision[['Duration to accept offer','DOJ Extended','Percent difference CTC','Joining Bonus','Offered band']]
y = job_decision['Status']


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

from sklearn.linear_model import LogisticRegression
logModel = LogisticRegression()
logModel.fit(X_train,y_train)

#predicting test data values
y_pred = logModel.predict(X_test)
print(y_pred)

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

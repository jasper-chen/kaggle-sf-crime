
# coding: utf-8

# In[76]:

import sklearn
sklearn.__version__


# In[77]:

from sklearn.cross_validation import train_test_split
import pandas as pd
from time import strptime
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


# In[78]:

X = pd.DataFrame(train.drop('Category',1))
y = pd.DataFrame(train['Category'])


# In[79]:

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .4)


# In[290]:

def convert_time(raw_datetime):
    """
    Classifies time when crime occured into groups.
    
    Args:
        raw_datetime (str): datetime
        
    Returns:
        int: a value which corresponds to a time range 
    """
    hour = strptime(x, "%Y-%m-%d %H:%M:%S").tm_hour
    if hour >= 21 or hour <= 6: #9pm - 6am
        return 0
    elif hour <= 12 and hour > 6: #6am - 12pm 
        return 1
    elif hour < 17 and hour > 12: #12pm - 5pm
        return 2
    else:
        return 3
    return hour


# In[267]:

def convert_month(raw_datetime):
    """
    Converts datetimes into months.
    
    Args:
        raw_datetime (str): datetime
        
    Returns:
        int: a month 
    """    
    month = strptime(x, "%Y-%m-%d %H:%M:%S").tm_mon
#    if month >= 1 and month <= 3:
#        return 0
#    elif month >= 4 and month <= 6:
#        return 1
#    elif month >= 7 and month <= 9:
#        return 2
#    else: 
#        return 3
    return month


# In[268]:

df = pd.DataFrame(X_train)


# In[269]:

df.columns = ['Dates','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']


# In[270]:

df['Hours'] = pd.DataFrame(df.Dates).applymap(convert_time)


# In[271]:

def convertDow(x):
    #clustering similar days with one another 
    days = {'Monday' : 1, 'Tuesday': 1, 'Wednesday': 1, 'Thursday': 1, 'Friday': 0, 'Saturday': 1, 'Sunday': 2}
    return days[x]

def convertDistrict(x):
    districts = {'MISSION' : 0, 'CENTRAL': 1, 'TARAVAL': 2, 'INGLESIDE': 3, 'TENDERLOIN': 4, 'BAYVIEW': 5, 'SOUTHERN': 6, 'NORTHERN': 7, 'PARK': 8, 'RICHMOND': 9}
    return districts[x]


# In[272]:

df['Season'] = pd.DataFrame(df.Dates).applymap(convert_month)


# In[273]:

df['DayOfWeek'] = pd.DataFrame(df.DayOfWeek).applymap(convertDow)


# In[274]:

df['PdDistrict'] = pd.DataFrame(df.PdDistrict).applymap(convertDistrict)


# In[275]:

def convert(x):
    crimes = {
            'ARSON' : 0,'ASSAULT': 1,'BAD CHECKS': 2,'BRIBERY': 3,'BURGLARY': 4,'DISORDERLY CONDUCT': 5,'DRIVING UNDER THE INFLUENCE': 6,
             'DRUG/NARCOTIC': 7,'DRUNKENNESS': 8,'EMBEZZLEMENT': 9,'EXTORTION': 10,'FAMILY OFFENSES': 11,'FORGERY/COUNTERFEITING': 12,
             'FRAUD': 13,'GAMBLING': 14,'KIDNAPPING': 15,'LARCENY/THEFT': 16,'LIQUOR LAWS': 17,'LOITERING': 18,'MISSING PERSON': 19,'NON-CRIMINAL': 20,
             'OTHER OFFENSES': 21,'PORNOGRAPHY/OBSCENE MAT': 22,'PROSTITUTION': 23,'RECOVERED VEHICLE': 24,'ROBBERY': 25,'RUNAWAY': 26,'SECONDARY CODES': 27,
             'SEX OFFENSES FORCIBLE': 28,'SEX OFFENSES NON FORCIBLE': 29,'STOLEN PROPERTY': 30,'SUICIDE': 31,'SUSPICIOUS OCC': 32,'TREA': 33,'TRESPASS': 34,
             'VANDALISM': 35,'VEHICLE THEFT': 36,'WARRANTS':37,'WEAPON LAWS': 38}
    return crimes[x]
df_y = pd.DataFrame(y_train).applymap(convert)


# In[276]:

from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


# In[277]:

#rf_model = RandomForestClassifier(n_estimators=100)


# In[278]:

enc = OneHotEncoder()


# In[279]:

pipeline = Pipeline([
    ('sgd',SGDClassifier())
])

parameters = {
    'sgd__alpha': (0.00001, 0.000001),
    'sgd__n_iter': (5,10,50,80)
}

#logistic = linear_model.LogisticRegressionCV(Cs=np.logspace(-4, 4, 3))
naivebayes = MultinomialNB(alpha=2.80)
#grid_search = GridSearchCV(pipeline,parameters, verbose=2)
#sgd = SGDClassifier(loss='log', alpha=0.00001, n_iter=50)


# In[280]:

new = pd.concat([df.Season,df.Hours,df.DayOfWeek, df.PdDistrict], axis=1)
#new = pd.concat([df.Hours, df.PdDistrict], axis=1)


# In[281]:

encoded = enc.fit_transform(new).toarray()


# In[282]:

#logistic_model = logistic.fit(new,df_y)
#logistic_model = logistic.fit(encoded,df_y)
bayes_model = naivebayes.fit(encoded,df_y)
#sgd_model = sgd.fit(encoded,df_y.values)
#rf_model = rf.fit(encoded,df_y)


# In[283]:

from sklearn.metrics import log_loss
#from sklearn.grid_search import GridSearchCV


# In[284]:

dft = pd.DataFrame(X_test)


# In[285]:

#dft.columns = ['Dates','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']
#dft['DayOfWeek'] = pd.DataFrame(dft.DayOfWeek).applymap(convertDow)
#dft['PdDistrict'] = pd.DataFrame(dft.PdDistrict).applymap(convertDistrict)
#dft['Hours'] = pd.DataFrame(dft.Dates).applymap(convert_time)
#dft['Season'] = pd.DataFrame(dft.Dates).applymap(convert_month)


# In[286]:

new_t = pd.concat([dft.Season,dft.Hours, dft.DayOfWeek, dft.PdDistrict], axis=1)
dfty = pd.DataFrame(y_test).applymap(convert)


# In[287]:

def logmap(x):
    #return logistic_model.predict_proba(x)
    return bayes_model.predict_proba(x)
    #return sgd_model.predict_proba(x)
    #return rf_model.predict_proba(x)
    #return grid_search_model.predict_proba(x)


# In[288]:

#estimates = logmap(new_t)
estimates = logmap(enc.transform(new_t).toarray())


# In[289]:

print "%.6f" % log_loss(dfty,estimates)

#10 bayes with 9 hour no class: 2.595
#9 bayes with 8 hour reclassification: 2.597
#8 bayes with 5 dayofweek reclassification: 2.600
#5 bayes with 4 one hot encoding : 2.602
#7 randomforest : 2.646
#6 logistic with 2 one hot encoding: 2.649
#2 logistic (hours,dayofweek,district): 2.673
#1 logistic (hours,district): 2.675
#4 bayes with (hours,dayofweek,district): 2.678
#3 bayes with (hours,district): :2.679


# #Test

# In[366]:

dfr = pd.DataFrame(test)
dfr['Hours'] = pd.DataFrame(dfr.Dates).applymap(convert_time)
dfr['DayOfWeek'] = pd.DataFrame(dfr.DayOfWeek).applymap(convertDow)
dfr['PdDistrict'] = pd.DataFrame(dfr.PdDistrict).applymap(convertDistrict)
dfr['Season'] = pd.DataFrame(dfr.Dates).applymap(convert_month)


# In[367]:

new_r = pd.concat([dfr.Season,dfr.Hours,dfr.DayOfWeek, dfr.PdDistrict], axis=1)


# In[368]:

enc = OneHotEncoder()
testbayes = MultinomialNB(alpha=1.0)
testcoded = enc.fit_transform(new_r).toarray()


# In[369]:

estimates_r = logmap(testcoded)


# In[370]:

Id = dfr['Id']
dfa= pd.DataFrame(estimates_r)
dfest = pd.concat([dfr.Id,dfa], axis=1)


# In[371]:

dfest.columns = ['Id','ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE',
             'DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING',
             'FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL',
             'OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES',
             'SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS',
             'VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS']

dfest.to_csv('results5.csv', index = False, columns=['Id','ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE',
             'DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING',
             'FRAUD','GAMBLING','KIDNAPPING','LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL',
             'OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES',
             'SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS',
             'VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS'])
#printing out predictions in csv format for submission to Kaggle competition


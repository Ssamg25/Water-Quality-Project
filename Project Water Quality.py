#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:#B9F8D3; color:#1A120B;">
#     <h1><center>**Group 3**</center>
#     </h1> </div>
# 
# ##### Name - 
# 1. **Shunottara Alhat**
# 
# # Project Water Quality

# ### Features Description 
# 
# 1. **ph Value**
# 
# 2. **Hardness**
# 
# 3. **Solids**
# 
# 4. **Chloramines**
# 
# 5. **Sulfate**
# 
# 6. **Conductivity**
# 
# 7. **Organic_carbon**
# 
# 8. **Trihalomethanes**
# 
# 9. **Turbidity**
# 
# 10. **Potability**

# ## Import Libraries 

# In[1]:


# Basic Libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# Visualizations Libraries
import plotly.express as pex
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
import xgboost as xgb

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_auc_score,roc_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import  DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC


# ## Import Data 

# In[2]:


Water = pd.read_excel("water_potability.xlsx")
Water


# <div style="background-color:#B9F8D3; color:#1A120B;">
#     <h1><center>**Exploratory Data Analysis**</center>
#                     <center> and
#     <center>**Visualizations**</center> </h1> </div>

# #### How Many Samples of Water are Potable 

# In[3]:


df = pd.DataFrame(Water["Potability"].value_counts())
fig = pex.pie(df, values = 'Potability', names = ['Not Potable','Potable'], hole = 0.4, opacity = 0.6,
             labels = {'label':'Potability','Potability':'No. of Samples'} , template='plotly_dark')
fig.add_annotation(text = 'We can resample the data <br> to get a balanced dataset',
                  x = 1.2, y = 0.9, showarrow = False, font_size = 12, opacity = 0.7, font_family = 'monospace')
fig.add_annotation(text = 'Potability', x = 0.5, y = 0.5, showarrow = False, font_size = 14, opacity = 0.7,
                  font_family = 'monospace')
fig.update_layout(font_family = 'monospace', title = dict(text = 'Q). How Many Samples of Water are Potable?',
                                                          x = 0.47, y = 0.98),
                 legend = dict(x = 0.37, y = -0.05, orientation = 'h', traceorder = 'reversed'),
                 hoverlabel = dict(bgcolor = 'white'))
fig.update_traces(textposition = 'outside', textinfo = 'percent+label')

fig.show()


# #### Hardness Distribution 

# In[4]:


fig = pex.histogram(Water, x ='Hardness',y = Counter(Water['Hardness']),color='Potability',template='plotly_dark',
                  marginal='box',opacity=0.7,nbins=100,
                  barmode='group',histfunc='count')

fig.add_vline(x = 151, line_width=1, line_dash='dot',opacity=0.7)
fig.add_vline(x = 301, line_width=1, line_dash='dot',opacity=0.7)
fig.add_vline(x = 76, line_width=1, line_dash='dot',opacity=0.7)

fig.add_annotation(text='<76 mg/L is<br> considered soft',x=40,y=130,showarrow=False,font_size=9)
fig.add_annotation(text='Between 76 and 150<br> (mg/L) is<br>moderately hard',x=113,y=130,showarrow=False,font_size=9)
fig.add_annotation(text='Between 151 and 300 (mg/L)<br> is considered hard',x=250,y=130,showarrow=False,font_size=9)
fig.add_annotation(text='>300 mg/L is<br> considered very hard',x=340,y=130,showarrow=False,font_size=9)

fig.update_layout(
    font_family='monospace',
    title=dict(text='Hardness Distribution',x=0.53,y=0.95,
               font=dict(size=20)),
    xaxis_title_text='Hardness (mg/L)',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,borderwidth=0,tracegroupgap=5),
    bargap=0.3,
)
fig.show()


# #### PH Level Distribution 

# In[5]:


fig = pex.histogram(Water,x='ph',y =Counter(Water['ph']),color='Potability',template='plotly_dark',
                  marginal='box',opacity=1,nbins=100,
                  barmode='group',histfunc='count')

fig.add_vline(x=7, line_width=1, line_dash='dot',opacity=0.7)

fig.add_annotation(text='<7 is Acidic',x=4,y=70,showarrow=False,font_size=10)
fig.add_annotation(text='>7 is Basic',x=10,y=70,showarrow=False,font_size=10)


fig.update_layout(
    font_family='monospace',
    title=dict(text='PH Level Distribution',x=0.5,y=0.95,
               font=dict(size=20)),
    xaxis_title_text='PH Level',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,borderwidth=0,tracegroupgap=5),
    bargap=0.3,
)
fig.show()


# #### TDS (Total Dissolved Solids)

# In[6]:


fig = pex.histogram(Water,x='Solids',y=Counter(Water['Solids']),color='Potability',template='plotly_dark',
                  marginal='box',opacity = 1,nbins=100,
                  barmode='group',histfunc='count')

fig.update_layout(
    font_family='monospace',
    title=dict(text='Distribution Of Total Dissolved Solids',x=0.5,y=0.95,
               font=dict(size=20)),
    xaxis_title_text='Dissolved Solids (ppm)',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,borderwidth=0,tracegroupgap=5),
    bargap=0.3,
)
fig.show()


# #### Chlorimines Distribution 

# In[7]:


fig = pex.histogram(Water,x='Chloramines',y=Counter(Water['Chloramines']),color='Potability',template='plotly_dark',
                  marginal='box',opacity=0.7,nbins=100,
                  barmode='group',histfunc='count')

fig.add_vline(x=4, line_width=1, line_dash='dot',opacity=0.7)

fig.add_annotation(text='<4 ppm is considered<br> safe for drinking',x=1.8,y=90,showarrow=False)

fig.update_layout(
    font_family='monospace',
    title=dict(text='Chloramines Distribution',x=0.53,y=0.95,
               font=dict(size=20)),
    xaxis_title_text='Chloramines (ppm)',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,borderwidth = 0,tracegroupgap = 5),
    bargap=0.3,
)
fig.show()


# #### Sulfate Distribution 

# In[8]:


fig = pex.histogram(Water,x='Sulfate',y = Counter(Water['Sulfate']),color='Potability',template='plotly_dark',
                  marginal='box',opacity=0.7, nbins=100,
                  barmode='group',histfunc='count')

fig.add_vline(x = 250, line_width=1, line_dash='dot',opacity=0.7)

fig.add_annotation(text='<250 mg/L is considered<br> safe for drinking',x = 175,y = 90,showarrow = False)

fig.update_layout(
    font_family='monospace',
    title=dict(text='Sulfate Distribution',x=0.53,y=0.95, font=dict(size=20)),
    xaxis_title_text='Sulfate (mg/L)',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96, borderwidth = 0,tracegroupgap = 5),
    bargap=0.3,
)
fig.show()


# #### Conductivity Distribution 

# In[9]:


fig = pex.histogram(Water,x = 'Conductivity', y = Counter(Water['Conductivity']),color='Potability',template='plotly_dark',
                  marginal='box',opacity = 0.7,nbins=100,barmode='group',histfunc='count')

fig.add_annotation(text='The Conductivity range <br> is safe for both (200-800),<br> Potable and Non-Potable water',
                   x=600,y=90,showarrow=False)

fig.update_layout(
    font_family='monospace',
    title=dict(text='Conductivity Distribution',x = 0.5,y = 0.95,
               font=dict(size=20)),
    xaxis_title_text='Conductivity (μS/cm)',
    yaxis_title_text='Count',
    legend=dict(x = 1, y = 0.96, borderwidth = 0,tracegroupgap = 5),
    bargap=0.3,
)
fig.show()


# #### Organic Carbon Distribution 

# In[10]:


fig = pex.histogram(Water, x = 'Organic_carbon', y = Counter(Water['Organic_carbon']), color = 'Potability',
                    template='plotly_dark', marginal='box',opacity=0.7,nbins=100,
                  barmode='group',histfunc='count')

fig.add_vline(x=10, line_width=1, line_dash='dot',opacity=0.7)

fig.add_annotation(text='Typical Organic Carbon<br> level is upto 10 ppm',x=5.3,y=110,showarrow=False)

fig.update_layout(
    font_family='monospace',
    title=dict(text='Organic Carbon Distribution',x=0.5,y=0.95,
               font=dict(size=20)),
    xaxis_title_text='Organic Carbon (ppm)',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,borderwidth=0,tracegroupgap=5),
    bargap=0.3,
)
fig.show()


# #### Trihalomethanes Distribution

# In[11]:


fig = pex.histogram(Water, x ='Trihalomethanes',y = Counter(Water['Trihalomethanes']),color='Potability',template='plotly_dark',
                  marginal='box',opacity=0.7,nbins=100,barmode='group',histfunc='count')

fig.add_vline(x=80, line_width=1, line_dash='dot',opacity=0.7)

fig.add_annotation(text='Upper limit of Trihalomethanes<br> level is 80 μg/L',x=115,y=90,showarrow=False)

fig.update_layout(
    font_family='monospace',
    title=dict(text='Trihalomethanes Distribution',x=0.5,y=0.95,
               font=dict(size=20)),
    xaxis_title_text='Trihalomethanes (μg/L)',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96,borderwidth=0,tracegroupgap = 5),
    bargap=0.3,
)
fig.show()


# #### Turbidity Distribution 

# In[12]:


fig = pex.histogram(Water, x='Turbidity', y = Counter(Water['Turbidity']),color='Potability',template='plotly_dark',
                  marginal='box', opacity=0.7,nbins=100, barmode='group', histfunc='count')

fig.add_vline(x=5, line_width = 1, line_dash='dot',opacity=0.7)

fig.add_annotation(text='<5 NTU Turbidity is<br> considered safe',x=6,y=90,showarrow=False)

fig.update_layout(
    font_family='monospace',
    title=dict(text='Turbidity Distribution',x=0.5,y=0.95,
               font=dict(size=20)),
    xaxis_title_text='Turbidity (NTU)',
    yaxis_title_text='Count',
    legend=dict(x=1,y=0.96, borderwidth=0, tracegroupgap=5),
    bargap=0.3,
)
fig.show()


# #### Correlation 

# In[13]:


Water.corr()


# #### Scatter Plot Matrix helps in finding out the correlation between all the features. 

# In[113]:


fig = pex.scatter_matrix(Water,Water.drop('Potability',axis=1),height=600,width=800,template='ggplot2',opacity=0.7,
                        color ='Potability', symbol ='Potability')

fig.update_layout(font_family='monospace',font_size=10,
                  coloraxis_showscale=False,
                 legend=dict(x=0.02,y=1.07),
                 title=dict(text='Scatter Plot Matrix Between Features',x=0.5,y=0.97,
                   font=dict(size=20)))
fig.show()


# In[15]:


cor = Water.drop('Potability',axis = 1).corr()
cor


# In[ ]:





# #### Heatmap to visualize the correlation

# In[16]:


fig = pex.imshow(cor,height=800,width=800,template='plotly_dark')

fig.update_layout(font_family='monospace',
                title=dict(text='Correlation Heatmap',x=0.5,y=0.93,
                             font=dict(size=24)),
                coloraxis_colorbar=dict(len=0.85,x=1.1) 
                 )

fig.show()


# ###  Info 

# In[17]:


Water.info()


# ### Missing Value 

# In[18]:


Water.isnull().sum()


# #### There are some Missing Values in PH, Sulfate, Trihalomethanes. 

# In[19]:


Water[Water['Potability']==0].describe()


# In[20]:


Water[Water['Potability']==1].describe()


# In[21]:


Water.drop(["Trihalomethanes"],inplace=True, axis =1)


# In[22]:


Water.fillna(Water.mean(),inplace=True)
Water.head()


# In[23]:


Water[Water['Potability']==0][['ph','Sulfate']].median()


# In[24]:


Water[Water['Potability']==1][['ph','Sulfate']].median()


# #### We can see that the difference between the mean and median values of Potable and Non-Potable Water is minimal. So we use the overall median of the feature to impute the values

# In[25]:


Water['ph'].fillna(value = Water['ph'].median(),inplace = True)
Water['Sulfate'].fillna(value = Water['Sulfate'].median(),inplace = True)


# In[26]:


Water.isnull().sum()


# In[27]:


Water.Potability.value_counts()


# #### Duplicate

# In[28]:


Water.duplicated().sum()


# #### Shape 

# In[29]:


Water.shape


# #### DataTypes 

# In[30]:


Water.dtypes


# <div style="background-color:#B9F8D3; color:#1A120B;">
#     <h1><center>**Splitting**</center>
#     </h1></div>

# In[31]:


X = Water.drop(columns="Potability")
Y = Water.Potability


# In[32]:


# Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.80, random_state=42, stratify=Y)


# <div style="background-color:#B9F8D3; color:#1A120B;">
#     <h1><center>**Model Building**</center></h1></div>

# ### Smote 

# In[33]:


sm = SMOTE(random_state = 42)
X_train, Y_train = sm.fit_resample(X_train, Y_train)


# In[34]:


Y_train.value_counts()


# In[35]:


Y_test.value_counts()


# In[36]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Logistic Regression 

# In[37]:


logit = LogisticRegression()
logit.fit(X_train, Y_train)


# In[38]:


logit.score(X_train, Y_train), logit.score(X_test, Y_test)


# In[39]:


pred1 = logit.predict(X_train)
pred_test1 = logit.predict(X_test)

print(confusion_matrix(Y_test, pred_test1))

print(classification_report(Y_test, pred_test1))


# ### Decision Tree 

# In[40]:


model1 = DecisionTreeClassifier(max_depth=5)
model1.fit(X_train, Y_train) 


# In[41]:


model1.score(X_train, Y_train), model1.score(X_test, Y_test)


# In[42]:


pred2 = model1.predict(X_train)
pred_test2 = model1.predict(X_test)

print(confusion_matrix(Y_test, pred_test2))

print(classification_report(Y_test, pred_test2))


# ### KFold 

# In[43]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_dt_train = cross_val_score(estimator=model1, X=X_train, y=Y_train, cv=k_fold)

cvscore_dt_train, cvscore_dt_train.std(), cvscore_dt_train.mean()


# In[44]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_dt_test = cross_val_score(estimator=model1, X=X_test, y=Y_test, cv=k_fold)

cvscore_dt_test, cvscore_dt_test.std(), cvscore_dt_test.mean()


# ### Random Forest 

# In[45]:


rf= RandomForestClassifier()
param_grid = [
{'n_estimators': [10, 25, 50,75,100], 
 'max_depth': [2, 3, 4, 5, 8, 10, 12, 15], 
 'bootstrap': [True, False],
 'max_features':["sqrt","auto","log2", 0.2, None]}
]

random_search = RandomizedSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
random_search.fit(X_train, Y_train)


# In[46]:


random_search.best_params_, random_search.best_estimator_


# In[47]:


rf= RandomForestClassifier(bootstrap=False, max_depth=15, max_features='log2',
                       n_estimators=50)
rf.fit(X_train, Y_train)


# In[48]:


rf.score(X_train, Y_train), rf.score(X_test, Y_test)


# In[49]:


pred3 = rf.predict(X_train)
pred_test3 = rf.predict(X_test)

print(confusion_matrix(Y_test, pred_test2))

print(classification_report(Y_test, pred_test2))


# ### XGBClassifier 

# In[50]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15, 17, 20],
 "min_child_weight" : [0.25, 0.05, 0.5, 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 "n_estimators"     : [50, 100, 200, 150, 250, 300]  
}

xgb_clf = XGBClassifier()

random_search = RandomizedSearchCV(xgb_clf, param_distributions=params, n_iter=5, scoring='roc_auc',
                                 n_jobs=-1, cv=5, verbose=3, random_state=42)
random_search.fit(X_train, Y_train)


# In[51]:


random_search.best_params_, random_search.best_estimator_


# In[52]:


xgb_clf_1 = XGBClassifier(n_estimators=250, max_depth=17, min_child_weight=1,learning_rate= 0.05)
xgb_clf_1.fit(X_train, Y_train)
predictions = xgb_clf_1.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions))

print("Classification Report")
print(classification_report(Y_test, predictions))


# In[53]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_train = cross_val_score(estimator=xgb_clf_1, X=X_train, y=Y_train, cv=k_fold)

cvscore_train, cvscore_train.std(), cvscore_train.mean()


# In[54]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_test = cross_val_score(estimator=xgb_clf_1, X=X_test, y=Y_test, cv=k_fold)

cvscore_test, cvscore_test.std(), cvscore_test.mean()


# <div style="background-color:#B9F8D3; color:#1A120B;">
#     <h1><center>**Base Model**</center>
#     </h1></div>

# In[55]:


water = pd.read_csv("water.csv", index_col="Unnamed: 0")
water


# In[56]:


water.Potability.value_counts()


# In[57]:


water.describe()


# ### Splitting 

# In[58]:


X = water.drop(columns="Potability")
Y = water.Potability
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.80, random_state=42, stratify=Y)


# ### Standard Scaler 

# In[59]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ### Logistic Regression 

# In[60]:


logit = LogisticRegression()
logit.fit(X_train, Y_train)


# In[61]:


logit.score(X_train, Y_train), logit.score(X_test, Y_test)


# In[62]:


pred1 = logit.predict(X_train)
pred_test1 = logit.predict(X_test)

print(confusion_matrix(Y_test, pred_test1))

print(classification_report(Y_test, pred_test1))


# #### Decision Tree 

# In[63]:


model1 = DecisionTreeClassifier(max_depth=5)
model1.fit(X_train, Y_train) 


# In[64]:


model1.score(X_train, Y_train), model1.score(X_test, Y_test)


# In[65]:


pred2 = model1.predict(X_train)
pred_test2 = model1.predict(X_test)

print(confusion_matrix(Y_test, pred_test2))

print(classification_report(Y_test, pred_test2))


# ### Kfold 

# In[66]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_train = cross_val_score(estimator=model1, X=X_train, y=Y_train, cv=k_fold)

cvscore_train, cvscore_train.std(), cvscore_train.mean()


# In[67]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_test = cross_val_score(estimator=model1, X=X_test, y=Y_test, cv=k_fold)

cvscore_test, cvscore_test.std(), cvscore_test.mean()


# ### Random Forest 

# In[68]:


rf= RandomForestClassifier()
param_grid = [
{'n_estimators': [10, 25, 50,75,100], 
 'max_depth': [2, 3, 4, 5, 8, 10, 12, 15], 
 'bootstrap': [True, False],
 'max_features':["sqrt","auto","log2", 0.2, None]}
]

random_search = RandomizedSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
random_search.fit(X_train, Y_train)


# In[69]:


random_search.best_params_, random_search.best_estimator_


# In[70]:


rf= RandomForestClassifier(bootstrap=False, max_depth=8, max_features='log2',
                       n_estimators=75)
rf.fit(X_train, Y_train)


# In[71]:


rf.score(X_train, Y_train), rf.score(X_test, Y_test)


# In[72]:


pred3 = rf.predict(X_train)
pred_test3 = rf.predict(X_test)

print(confusion_matrix(Y_test, pred_test2))

print(classification_report(Y_test, pred_test2))


# #### XGBClassifier 

# In[73]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15, 17, 20],
 "min_child_weight" : [0.25, 0.05, 0.5, 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 "n_estimators"     : [50, 100, 200, 150, 250, 300]  
}

xgb_clf=XGBClassifier()

random_search=RandomizedSearchCV(xgb_clf, param_distributions=params, n_iter=5, scoring='roc_auc',
                                 n_jobs=-1, cv=5, verbose=3, random_state=42)
random_search.fit(X_train, Y_train)


# In[74]:


random_search.best_params_, random_search.best_estimator_


# In[75]:


xgb_clf_1 = XGBClassifier(n_estimators=250, max_depth=17, min_child_weight=1,learning_rate= 0.05)
xgb_clf_1.fit(X_train, Y_train)
predictions = xgb_clf_1.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions))

print("Classification Report")
print(classification_report(Y_test, predictions))


# #### KFold 

# In[76]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_train = cross_val_score(estimator=xgb_clf_1, X=X_train, y=Y_train, cv=k_fold)

cvscore_train, cvscore_train.std(), cvscore_train.mean()


# In[77]:


k_fold = KFold(n_splits=5, shuffle=True, random_state=123)
cvscore_dt_test = cross_val_score(estimator=xgb_clf_1, X=X_test, y=Y_test, cv=k_fold)

cvscore_dt_test, cvscore_dt_test.std(), cvscore_dt_test.mean()


# ## XGBoost  

# In[78]:


import xgboost as xgb

xgb.plot_importance(xgb_clf_1)
plt.figure(figsize = (20, 25))
plt.show()


# In[79]:


X.corr()


# In[80]:


svm = SVC(random_state = 42)
svm.fit(X_train, Y_train)


# In[81]:


predictions = svm.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions))

print("Classification Report")
print(classification_report(Y_test, predictions))


# In[82]:


param_grid={
    'C':[0.1,0.5,1,2,3,4,5],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'degree':[3,4,5,6,7]
    
}
grid_SVC=GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_SVC.fit(X_train,Y_train)
print(grid_SVC.best_params_,grid_SVC.best_score_)


# In[83]:


svm_1 = SVC(random_state = 42, C=4, kernel='rbf')
svm_1.fit(X_train, Y_train)


# In[84]:


predictions = svm_1.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions))

print("Classification Report")
print(classification_report(Y_test, predictions))


# ### TensorFlow 

# In[85]:


#pip install tensorflow


# In[86]:


from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, Adagrad, RMSprop, SGD
from tensorflow.keras.layers import Activation
from sklearn.metrics import accuracy_score


# In[87]:


X_val, X_test, Y_val, Y_test = train_test_split(X_test,Y_test, test_size = 0.5, random_state = 42)


# In[88]:


model = models.Sequential()

model.add(layers.Dense(16, input_shape=(9,)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(layers.Dense(32))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(layers.Dense(16))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(layers.Dense(1))
model.add(Activation("sigmoid"))


# In[89]:


opt = Adam(learning_rate=0.001)

model.compile(loss="binary_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])


# In[90]:


model.summary()


# In[91]:


tf.random.set_seed(0)

history = model.fit(X_train,
                    Y_train,
                    epochs=300,
                    batch_size=32,
                    validation_data=(X_val, Y_val),
                   )


# In[92]:


score = model.evaluate(X_test, Y_test, verbose=1)

print("Test Error", score[0])
print("Test accuracy", score[1])


# #### XGBClassifier Final 

# In[93]:


xgb_clf_final = XGBClassifier(n_estimators=250, max_depth=17, min_child_weight=1,learning_rate= 0.05)
xgb_clf_final.fit(X_train, Y_train)
predictions_test = xgb_clf_final.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions_test))

print("Classification Report")
print(classification_report(Y_test, predictions_test))


# In[94]:


predictions_val = xgb_clf_final.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(Y_val, predictions_val))

print("Classification Report")
print(classification_report(Y_val, predictions_val))


# <div style="background-color:#B9F8D3; color:#1A120B;">
#     <h1><center>**Final Model**</center>
#     </h1></div>

# In[95]:


water = pd.read_csv("water.csv", index_col="Unnamed: 0")


# #### Splitting 

# In[96]:


X = water.drop(columns="Potability")
Y = water.Potability
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=.80, random_state=42, stratify=Y)


# In[97]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 13, 14, 15, 17, 20],
 "min_child_weight" : [0.25, 0.05, 0.5, 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
 "n_estimators"     : [50, 100, 200, 150, 250, 300] 
}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_clf=XGBClassifier()

random_search=RandomizedSearchCV(xgb_clf, param_distributions=params, n_iter=5, scoring='roc_auc',
                                 n_jobs=-1, cv=kfold, verbose=3, random_state=42)
random_search.fit(X_train, Y_train)


# In[98]:


random_search.best_params_, random_search.best_estimator_


# In[99]:


model = XGBClassifier(n_estimators= 250, min_child_weight= 0.5, max_depth= 20, learning_rate= 0.2, gamma= 0.3,
                      colsample_bytree= 0.7, random_state=42)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(Y_test, predictions))

print("Classification Report")
print(classification_report(Y_test, predictions))


# #### Features Importance 

# In[100]:


xgb.plot_importance(model)
plt.figure(figsize = (20, 25))
plt.show()


# In[101]:


X


# In[102]:


Y


# In[ ]:





# In[103]:


water = pd.read_csv("water.csv", index_col="Unnamed: 0")


# In[104]:


X = water.drop(columns=["Potability","Turbidity","Conductivity"])
Y = water.Potability


# In[105]:


model = XGBClassifier(n_estimators= 250, min_child_weight= 0.5, max_depth= 20, learning_rate= 0.2, gamma= 0.3,
                      colsample_bytree= 0.7, random_state=42)
model.fit(X, Y)


# In[106]:


import pickle
with open(file="Final_model.pkl", mode="wb") as f:
    pickle.dump(model, f)


# In[107]:


X


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
plt.rcParams['figure.figsize'] = (15,9)


# In[2]:


# Import dataset.

data = pd.read_excel('C:\\Users\\JAHFAITH IROKANULO\\Downloads\\Salary dataset response.xlsx')
data.head()


# In[3]:


# Data size.

data.shape, data.size


# In[4]:


data.columns


# In[5]:


# Drop columns that are not needed.

data = data.drop(['Unnamed: 0', 'Timestamp', 'Company', 'Country/Location'], axis=1)
data.shape


# In[6]:


data = data.rename(columns={'Degree ':'Degree'})


# In[7]:


data.isna().sum()


# In[8]:


# Drop the data points with missing salary values.

data = data.dropna()
data.shape


# In[ ]:





# In[9]:


# Let's see the distribution of the target variable -Monthly Salary.

data['Monthly Salary'].value_counts()


# In[10]:


# Visualizing the distribution.

data['Monthly Salary'].value_counts().plot.bar()


# In[11]:


# Representation of countries in each salary scale. 
cs = data.groupby('Monthly Salary')['Company location'].unique()
cs


# The above analysis shows the countries that are represented in each pay scale. Only Nigeria pays her developers between 401k-500k, Only Nigeria and Ghana pay below 201k. No American or European company paid below 201k.

# In[12]:


# Grouping the data by Company Location and Salary to see which locations have the highest Salary range.

data.groupby(['Company location', 'Monthly Salary']).count()


# 28 out of 31 Nigerians working remotely in America earn over 500k monthly.
# 
# 13 out of 15 Nigerians working remotely in Europe earn over 500k monthly.
# 
# In Nigeria, majority of the tech talents earn between 101k - 200k.

# In[13]:


# Encoding the target variable for future modeling.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
data['Monthly Salary'] = le.fit_transform(data['Monthly Salary'])


# In[14]:


data['Monthly Salary'].value_counts()


# In[ ]:





# In[15]:


# Company Location.

data['Company location'].value_counts()


# In[16]:


data['Company location'].replace({'Ghana':'Rest of africa'}, inplace=True)
data['Company location'].value_counts().plot.bar()


# In[ ]:





# In[17]:


# Title.
# Let's class these roles into fewer categories and put them in a new column.

def JobTitle(x):
    DS = ['Data analyst ', 'Business Analyst', 'Data science intern ', 'Data scientist', 
         'Data Analyst/Business Intelligence Analyst', 'Business Data Analyst', 'Data Analyst', 
          'Risk analyst', 'Senior Analyst', 'Analyst',  'Data Scientist ', 'Data Scientist', 'Research analyst', 
         'Financial Analyst ', 'Senior Data Scientist', ]
    
    SE = ['Full-stack Software Engineer', 'Junior Software Developer', 'Mid-Level Software Developer ', 
         'Software Engineering Lead', 'Software dev', 'Intermidiate Dev', 
         'Senior Engineer', 'Junior Software engineer', 'Junior Engineer ', 'Software developer ', 
          'Engineering Lead', 'Fullstack software engineer', 'Android Engineer ','Junior Software developer ', 
         'Software Enginer', 'Senior software engineer ', 'Lead Software Developer', 'Freelance software developer', 
         'Senior Software developer', 'Sofware developer', 'Android engineer', 'Mobile developer ', 
         'Software engineer', 'Jnr software developer ', 'Senior Software Engineer', 'Software developer',
          'Developer Advocate & Software Engineer ', 'Senior Software Engineer ', 'Software Engineer ',
         'Associate Software Engineer', 'Junior software engineer', 'iOS Developer', 'Software engineer ', 
          'Software Developer ', 'Software Developer', 'Senior software engineer', 'Software Consultant',
       'Android Developer', 'Software Engineer', 'Android Engineer', 'Junior Software Developer ', 
          'Developer', 'Senior Developer', 'Developer Advocate ', 'software developer ',]
    
    FD = ['Frontend Developer', 'Front-end Developer', 'Frontend web development', 'Frontend/mobile developer ', 
          'Frontend engineer', 'React Native Developer', 'Developer ', 'Frontend Development', 'Frontend developers',
       'Front End Engineer', 'React native developer', 'frontend', 'WordPress Developer', 'Frontend Engineer', 
         'Frontend/ mobile engineer ', 'Front end developer ', 'Front-End Developer(Intern)', 'Front End Developer', 
         'frontend developer']
    
    BD = ['Backend engineer.', 'Backend Developer ', 'Back end developer', 'Backend ', 'backend developer ', 
         'Mid level backend engineer ',  'Backend developer ', 'Backend Developer', 'Backend Engineer ',  
         'Backend Engineer', 'Backend developer']
    
    FS = ['FullStack Developer ',  'Full stack engineer', 'Full Stack', 'Fullstack', 'Full Stack', 
         'Fullstack Developer', 'Full stack (heavy on the backend) ', 'Full stack Web Developer ', 'Web developer ']
    
    UD = ['UX Designer', 'Ui/Ux designer', 'UX Designer ', 'Graphics Designer', 'Designer (Remote) ', 
          'Designer']
    
    DE = ['Devops', 'Data Engineer ', 'Software/DevOps Engineer ']
    
    PD = ['Associate Product Analyst', 'Product manager ', 'Product Designer ', 'Product analyst', 
         'Product Associate ', 'Project Manager', 'Program Manager', 'Product Designer']
    
    IT = ['IT Support ', 'Support ', 'Project support officer', 'Operations', 'Senior Systems Administrator', 
         'Technical support', 'CTO', 'support', 'Information Systems Manager',  'Quality assurance engineer']
    
    DM = ['Digital channels analyst',]
    
    aa = [  ]
    
    if x in DS:
        return 'Data Scientist/Analyst'
    elif x in SE:
        return 'Software Engineer'
    elif x in FD:
        return 'Frontend Developer'
    elif x in BD:
        return 'Backend Developer'
    elif x in FS:
        return 'Full Stack Developer'
    elif x in UD:
        return 'UI/UX/Graphics Designer'
    elif x in DE:
        return 'Data Engineer/Devops'
    elif x in PD:
        return 'Product Designer/Manager'
    elif x in IT:
        return 'IT/Operations'
    else:
        return 'Others'


# In[18]:


# Title is the parent/collective name of the different roles.

data['Title'] = data['Role'].map(JobTitle)
data.head()


# In[19]:


data['Title'].value_counts()


# In[20]:


title_chart = data.groupby('Title')['Title'].count()
title_chart.plot.pie()


# In[ ]:





# In[21]:


# Experience.
data['Experience'].value_counts().plot.bar()


# In[ ]:





# In[22]:


# Distribution of degrees across tech applicants.

sns.countplot(data['Degree'])


# In[23]:


data.loc[data['Degree'] == 'N']


# The table above shows that there are non degree techies with high paying jobs both in Nigeria and abroad.

# In[24]:


# Let's store all the rows where job title is 'Others' as Non_tech and count them.
Non_tech = data[data['Title'] == 'Others']
len(Non_tech)


# In[25]:


# Drop the rows using their index.
data = data.drop(Non_tech.index, axis=0)
data.shape


# In[26]:


data = data.drop(['Role'], axis=1)
data.head(3)


# In[ ]:





# In[27]:


# Label Encoding categorical features.
data.loc[:, ['Degree', 'Company location', 'Title']] = data.loc[:, ['Degree', 'Company location', 'Title']].apply(le.fit_transform)


# In[28]:


data.head(3)


# In[29]:


# 1 = Y, 0 = N.

data['Degree'].value_counts()


# In[30]:


# 0 = America, 1 = Europe, 2 = Nigeria, 3 = Rest of africa.

data['Company location'].value_counts()


# In[31]:


# 0=Backend, 1=Data Engineer/Devops, 2=Data Scientist/Analyst, 3=Frontend, 
# 4=Fullstack, 5=IT/Operations, 6=Products, 7=Software Engineer.

data['Title'].value_counts()


# In[ ]:





# In[32]:


# Correlation matrix

corr_matrix = data.corr()


# In[33]:


# Let's see how the features correlate with the target.
corr_matrix['Monthly Salary'].sort_values(ascending=False)


# In[34]:


corr_matrix


# In[ ]:





# In[35]:


# Splitting into X and Y axes.

X = data.drop(['Monthly Salary'], axis=1)
Y = data['Monthly Salary']

print('Shape of Features: ',X.shape)
print('')
print('Shape of Target: ',Y.shape)


# In[36]:


# Time To Model!

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import log_loss, f1_score


# In[37]:


fold = KFold(n_splits=5, shuffle=True, random_state=12)


# In[38]:


# Logistic Regression.

for train_index, test_index in fold.split(X, Y):
    # split into train and test.
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # train the model.
    reg = LogisticRegression(max_iter=1000, multi_class='multinomial', C=0.5)
    reg.fit(x_train, y_train)
    
    # making prediction.
    y_pred1 = reg.predict(x_test)
    
    # evaluate the model.
    f1 = f1_score(y_test, y_pred1, average='weighted')
    
    print('F1: ',f1)


# In[39]:


score = cross_val_score(reg, x_train, y_train, scoring='f1_weighted', cv=5)
avg_score = score.mean()
print('Mean Score: ',avg_score)


# In[40]:


# Comparing Actual and Predicted outputs of Y(Monthly Salary)

table = pd.DataFrame(y_test)
table['Predicted'] = y_pred1
table.head(10)


# In[ ]:





# In[41]:


# Random Forest.

for train_index, test_index in fold.split(X, Y):
    # split into train and test.
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # train the model.
    model = RandomForestClassifier(n_estimators=2000, max_depth=10, max_samples=20, random_state=12, max_features=4)
    model.fit(x_train, y_train)
    
    # making prediction.
    y_pred2 = model.predict(x_test)
    
    # evaluate the model.
    f1 = f1_score(y_test, y_pred2, average='weighted')
    
    print('F1: ',f1)


# In[42]:


score = cross_val_score(model, x_train, y_train, scoring='f1_weighted', cv=5)
avg_score = score.mean()
print('Mean Score: ',avg_score)


# In[ ]:





# In[43]:


# Gradient Boost.

from sklearn.ensemble import GradientBoostingClassifier

for train_index, test_index in fold.split(X, Y):
    # split into train and test.
    x_train, x_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # train the model.
    boost = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, max_depth=2)
    boost.fit(x_train, y_train)
    
    # making prediction.
    y_pred3 = boost.predict(x_test)
    
    # evaluate the model.
    f1 = f1_score(y_test, y_pred3, average='weighted')
    
    print('F1: ',f1)


# In[44]:


score = cross_val_score(boost, x_train, y_train, scoring='f1_weighted', cv=5)
avg_score = score.mean()
print('Mean Score: ',avg_score)


# In[ ]:





# In[45]:


# Serializing the Random Forest model.

import pickle


# In[46]:


# Saving the model.
pickle.dump(model, open('Pace_Model.pkl', 'wb'))
print('All kwaret Sir!')


# In[49]:


data.columns


# In[50]:


# Columns(Features) that were used to train the model.
x_train.columns


# In[53]:


# Saving the columns
pickle.dump(data.columns, open('Pace_Model_Columns.pkl', 'wb'))


# In[54]:


# Saving the data used.
pickle.dump(x_train.columns, open('Pace_Training_Features.pkl', 'wb'))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import pickle


# In[2]:


app = FastAPI(title= 'SALARY PREDICTOR')


# In[3]:


@app.get('/')
async def index():
    return {'text': 'Welcome TO MY SALARY APP'}


# In[4]:


@app.post('/predict')
def get_prediction(Degree: bool, Experience: int, Company_location: str, Title: str):
    
    model = pickle.load(open('Pace_Model.pkl', 'rb'))
    
    if Company_location == 'America':
        Company_location = 0
    elif Company_location == 'Europe':
        Company_location = 1
    elif Company_location == 'Nigeria':
        Company_location = 2
    elif Company_location == 'Africa':
        Company_location = 3
        
    if Title == 'Backend':
        Title = 0
    elif Title == 'Data Engineer':
        Title = 1
    elif Title == 'Devops':
        Title = 1
    elif Title == 'Data Scientist':
        Title = 2
    elif Title == 'Data Analyst':
        Title = 2
    elif Title == 'Frontend':
        Title = 3
    elif Title == 'Full stack':
        Title = 4
    elif Title == 'IT':
        Title = 5
    elif Title == 'Products':
        Title = 6
    elif Title == 'Software Engineer':
        Title = 7
    elif Title == 'Software Developer':
        Title = 7
    
    data = [[Degree, Experience, Company_location, Title]]
    prediction = model.predict(data)
    sal = prediction
    
    if sal==0:
        prediction='101k-200k'
    elif sal==1:
        prediction='201k-300k'
    elif sal==2:
        prediction='301k-400k'
    elif sal==3:
        prediction='401k-500k'
    elif sal==4:
        prediction='50k-100k'
    elif sal==5:
        prediction='<50k'
    elif sal==6:
        prediction='>500k'
    return {'Salary Range': prediction}



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)



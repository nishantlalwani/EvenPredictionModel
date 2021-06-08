#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pymysql


# In[15]:

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost:3306/{db}"
                       .format(user="root",
                               pw="aaryan007",
                               db="even"))
leads= pd.read_sql("select * from leads", engine.connect())
loan_purpose_list=leads['loan_purpose'].unique().tolist()


# Load the Random Forest CLassifier model

with open("train_model", "rb") as f:
    classifier = pickle.load(f) 

with open("encoder", "rb") as f:
    encoder = pickle.load(f)


# In[7]:


def convert_credit(key):
    dict={'limited':0,'unknown':0,'poor':1,'good':2,'fair':3,'excellent':4}
    return dict[key]
no_credit_func= lambda credit:1 if credit in ("limited",'unknown') else 0


# In[ ]:


app = Flask(__name__)

@app.route('/')
def home():
    credit_list=['limited','unknown','poor','good','fair','excellent']
    return render_template('index.html',credit=credit_list,loan=loan_purpose_list)


# In[ ]:


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        apr = list(request.form['APR'])
        requested = float(request.form['Requested_Amount'])
        annual_income = float(request.form['Annual_Income'])
        credit = str(request.form['Credit'])
        loan_purpose = str(request.form['Loan_Purpose'])
        
        
        credit_bucket=convert_credit(credit)
        no_credit=no_credit_func(credit)
        loan_purpose_encoded= encoder.transform([[loan_purpose]]).toarray()
	my_prediction ={}
        
	for i in apr:
		apr=int(i)
        	data = np.array([[apr, requested, annual_income, no_credit, credit_bucket]])
        	data=np.concatenate((data,loan_purpose_encoded),axis=1)
        	my_prediction[apr] = classifier.predict_proba(data)[0][1]
        print(my_prediction)
        
        return render_template('result.html', prediction=my_prediction)



# In[ ]:


if __name__ == '__main__':
    app.run(debug=True)







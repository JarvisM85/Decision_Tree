# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 09:30:30 2024

@author: sahil
"""

import pandas as pd
df = pd.read_csv("C:/DS2/1.3_Decision_Tree/salaries.csv")
df.head()
inputs = df.drop('salary_more_then_100k',axis='columns')
target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
inputs_n = inputs.drop(['company','job','degree'],axis='columns')








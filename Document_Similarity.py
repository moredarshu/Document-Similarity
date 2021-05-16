#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[22]:


# Define the documents
doc_trump = "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin"

doc_election = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election"

doc_putin = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career"

doc_sent = "He is a AI Engineer"


# In[23]:


# Converting to vector


doc = [doc_trump,doc_election,doc_putin,doc_sent]
cv = CountVectorizer(stop_words='english')
matrix = cv.fit_transform(doc)


# In[24]:


# Vector
matrix.toarray()


# In[25]:


# Features
cv.get_feature_names()


# In[26]:


df = pd.DataFrame(matrix.toarray(),columns=cv.get_feature_names(),index= ['doc_trump','doc_election','doc_putin','doc_sent'])
df


# In[27]:


print(cosine_similarity(df,df))


# In[29]:


model_filename = 'DocSimModel.pkl'
vec_filename = 'DocSimVec.pkl'
# joblib.dump(model_filename)
joblib.dump(cv,vec_filename)


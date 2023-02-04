#!/usr/bin/env python
# coding: utf-8

# # PCA OF CANCER DATA

# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd


# In[2]:


df=pd.read_csv("Cancer Data.csv")
df


# In[10]:


df["Cancer_Stage"]


# In[25]:


x=df.iloc[:,0:524]
y=df.iloc[:,-1]


# In[28]:


x


# In[ ]:


from sklearn.preprocessing import StandardScale


# In[29]:


x=StandardScaler().fit_transform(x)
x.shape


# In[8]:


from sklearn.decomposition import PCA
pca=PCA(n_components=19)
x=pca.fit_transform(x)

x=pd.DataFrame(x)

x


# In[43]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train=pd.DataFrame(x_train)
x_train.head()


# In[44]:


y_train=pd.DataFrame(y_train)
y_train


# In[45]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=42)
#fit the classifier to the train data
logreg.fit(x_train,y_train)


# In[46]:


y_pred =logreg.predict(x_test)


# In[47]:


logreg.score(x_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





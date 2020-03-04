
# coding: utf-8

# In[2]:


import sklearn


# In[3]:


from sklearn.datasets import load_breast_cancer


# In[22]:


from sklearn.model_selection import train_test_split


# In[10]:


from sklearn.naive_bayes import GaussianNB


# In[23]:


from sklearn.metrics import accuracy_score


# In[4]:


# load the dataset
data = load_breast_cancer()


# In[5]:


data


# In[6]:


# organize data better

label_names = data["target_names"]
labels = data["target"]
feature_names = data["feature_names"]
features = data["data"]


# In[7]:


label_names


# In[9]:


# split the data to train/test
train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33, random_state=42)


# In[11]:


# We use the Naive-Bayes algorithm 


# In[12]:


gnb_classifier = GaussianNB()


# In[13]:


model = gnb_classifier.fit(train, train_labels)


# In[14]:


model


# In[16]:


# make predictions
predictions = gnb_classifier.predict(test)
print(predictions)


# In[21]:


acc_score = accuracy_score(test_labels, predictions)
print(acc_score)


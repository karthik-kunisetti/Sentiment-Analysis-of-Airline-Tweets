#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[20]:


import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# # Data Loading

# In[10]:


airline_tweets=pd.read_csv("C:/Users/karth/Downloads/Tweets.csv")
airline_tweets


# # Basic Information

# In[11]:


airline_tweets.info()


# # Display top 5 rows

# In[12]:


airline_tweets.head()


# In[13]:


plot_size = plt.rcParams["figure.figsize"] 

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size 


# In[14]:


airline_tweets.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')


# In[ ]:


#It is evident from the output that for almost all the airlines, the majority of the tweets are negative, followed by neutral and positive tweets. Virgin America is probably the only airline where the ratio of the three sentiments is somewhat similar.


# In[15]:


airline_tweets.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])


# In[ ]:


#To view the average confidence level for the tweets belonging to three sentiment categories. 


# In[16]:


airline_sentiment = airline_tweets.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')


# In[17]:


import seaborn as sns

sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=airline_tweets)


# # Data Cleaning

# In[18]:


features = airline_tweets.iloc[:, 10].values
labels = airline_tweets.iloc[:, 1].values


# In[21]:


# We would keep the processed features here in this list
processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # Remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature) 

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)


# In[ ]:


### Representing Text with Numeric Vectors

Statistical algorithms use mathematics to train machine learning models. 

To make statistical algorithms work with text, we first have to convert text to numbers. 

To do so, three main approaches exist i.e. Bag-of-Words, TF-IDF and Word2Vec. 

In this section, we will discuss the bag of words and TF-IDF scheme.


# In[22]:


vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features = vectorizer.fit_transform(processed_features).toarray()


# 
# # Data Splitting into Training and Test Set

# In[23]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.2, random_state=0)


# In[25]:


X_train.shape


# # Build Machine Learning Models

# In[ ]:


#Now. lets build a number of machine laarning models and check their performance on the same test set.


# # First Model: Random Forest

# In[26]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=0)


# # Train the Random Forest Model

# In[27]:


rf.fit(X_train, y_train)


# # Random Forest Evaluation and prediction

# In[28]:


predictions = rf.predict(X_test)


# In[29]:


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


# # Second Model:Support Vector Machine

# In[30]:


# Create a SVM model
svm = SVC(kernel='linear', C=1.0)

# Train the model
svm.fit(X_train, y_train)

# Evaluate the model
predictions = svm.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


# # Third Model:Multinomial Naive Bayes

# In[31]:


# Create a Multinomial Naive-Bayes model
nb = MultinomialNB()

# Train the model
nb.fit(X_train, y_train)

# Evaluate the model
predictions = nb.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


# # Fourth Model:Logistic RegressionClassifier

# In[32]:


# Create a Logistic Regression Classification model
lr = LogisticRegression()

# Train the model
lr.fit(X_train, y_train)

# Evaluate the model
predictions = lr.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))


# In[33]:


#We observe that performance of all the models are quite competitive.

#The Random Forest model provides an accury of 75.99%.

#Where as the Support Vector machine reports an accuray of 78%.

#The accuracy of Naive Bayes is 75.81%.

#The Logistic Regression model gives 78.82% accuracy.


# In[ ]:





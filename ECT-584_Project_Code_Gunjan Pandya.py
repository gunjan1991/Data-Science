# -*- coding: utf-8 -*-
"""
Created on Sat Aug 3 19:59:12 2016

@author: gunjan

ECT-584 - Summer II Data Analysis Project - Gunjan Pandya (1547486)
"""

import nltk
import string, re as regexp
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
from sklearn.metrics import confusion_matrix as conmat
from sklearn import naive_bayes as nb,metrics, linear_model as lm
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neighbors import KNeighborsClassifier as kNN
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split as tts
#reading CSV file
data_frame1 = pd.read_csv('C:/Users/gunja/Downloads/GOP_REL_ONLY.csv',encoding='latin')

#To study class distirbution
sentiment_count_candidate = data_frame1.groupby('sentiment').size()
sentiment_count_candidate.plot(kind = "bar" ,  title = "Tweet Sentiment" , figsize=[6,6])

#To study candidate distirbution
candidate_count = data_frame1.groupby('candidate').size()
candidate_count.plot(kind = "bar" ,  title = "Candidates" , figsize=[6,6])

#reading CSV file
data_frame2 = pd.read_csv('C:/Users/gunja/Downloads/Airline-Sentiment-2-w-AA.csv',encoding='latin')

#To study class distirbution
sentiment_count_airline = data_frame2.groupby('airline_sentiment').size()
sentiment_count_airline.plot(kind = "bar" ,  title = "Tweet Sentiment" , figsize=[6,6])

#To study airline distirbution
airline_count = data_frame2.groupby('airline').size()
airline_count.plot(kind = "bar" ,  title = "Airlines" , figsize=[6,6])

#Start data preprocessing
stop_words = nltk.corpus.stopwords.words('english')
lemmatizer_list = nltk.stem.WordNetLemmatizer()
punctuation_list = list(string.punctuation)
stop_words_list = stop_words + punctuation_list +["rt"]

#Function to pre-process the tweets
def preProcessTweet(text):
        text = " ".join(text.split('#'))
        #Removing # and @
        text = regexp.sub(r'#([^\s]+)', r'\1', text)             
        text = regexp.sub('@[^\s]+',"",text)        
        #Replacing all occurences of *http*        
        text = regexp.sub('((www\.[^\s]+)|(https://[^\s]+))',"",text)
        text = regexp.sub("http\S+", "", text)
        text = regexp.sub("https\S+", "", text)
        text = regexp.sub('[\s]+', ' ', text)        
        #Replacing \
        text = text.replace('\"',"")
        #All tweet text to lower text
        text = text.lower()        
        #Stop-words removal          
        text  = " ".join([text_word for text_word in text.split(" ") if text_word not in stop_words_list])
        text  = " ".join([text_word for text_word in text.split(" ") if regexp.search('^[a-z]+$', text_word)])
        #Lemmatization
        text = " ".join([lemmatizer_list.lemmatize(text_word) for text_word in text.split(" ")])
        text = text.strip('\'"')
        
        return text

#Applying pre processing on text of the tweets
data_frame1['processedTweets'] = data_frame1.text.apply(preProcessTweet)
data_frame2['processedTweets'] = data_frame2.text.apply(preProcessTweet)

#Taking sentiments into one dataframe
categories_df1 = data_frame1.sentiment.unique()
categories_df2 = data_frame2.airline_sentiment.unique()
categories_df1  = categories_df1.tolist()
categories_df2  = categories_df2.tolist()

#For consistency in confusion matrix
categories_df1[0]='negative'
categories_df1[1]='neutral'
categories_df1[2]='positive'

categories_df2[0]='negative'
categories_df2[1]='neutral'
categories_df2[2]='positive'

#Taking X and Y variables into numpy array to use it in algorithms when creating models
x_variables1 = data_frame1.processedTweets.values
y_variables1 = data_frame1.sentiment.values

x_variables2 = data_frame2.processedTweets.values
y_variables2 = data_frame2.airline_sentiment.values

#Split into training and testing 80-20 split
x_variableTrain1, x_variableTest1, y_variableTrain1, y_variableTest1 = tts( x_variables1, y_variables1, test_size=0.2, random_state=1 )
x_variableTrain2, x_variableTest2, y_variableTrain2, y_variableTest2 = tts( x_variables2, y_variables2, test_size=0.2, random_state=1 )

#To see extracted features
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.3, sublinear_tf = True,stop_words = 'english')

x_train1_tfidf_matrix = tfidfvector.fit(x_variableTrain1)
idf = x_train1_tfidf_matrix._tfidf.idf_
wordDict=dict(zip(x_train1_tfidf_matrix.get_feature_names(), idf))

x_train2_tfidf_matrix = tfidfvector.fit(x_variableTrain2)
idf = x_train2_tfidf_matrix._tfidf.idf_
wordDict=dict(zip(x_train2_tfidf_matrix.get_feature_names(), idf))

#Multinomial Naive Bayes
start = time.time()

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix - Using TFIDF Vectorizer for feature extraction from text
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.3, sublinear_tf = True,stop_words = 'english')

#Model for Dataset 1
x_train1_tfidf_matrix = tfidfvector.fit(x_variableTrain1)
x_train1_tfidf_matrix = tfidfvector.transform(x_variableTrain1)
x_test1_tfidf_matrix = tfidfvector.transform(x_variableTest1)

multi_nb = nb.MultinomialNB(alpha=1,fit_prior=True)
mb = multi_nb.fit(x_train1_tfidf_matrix, y_variableTrain1)
multi_nb_pred1 = mb.predict(x_test1_tfidf_matrix)

accuracy_dataset1 = metrics.accuracy_score(y_variableTest1, multi_nb_pred1)
accuracy_dataset1*=100
print("Accuray on Dataset1 (Debate Data) Using Naive Bayes: %.2f" %(accuracy_dataset1) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest1, multi_nb_pred1),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest1, multi_nb_pred1, target_names=categories_df1))

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.3, sublinear_tf = True,stop_words = 'english')

#Model for Dataset 2
x_train2_tfidf_matrix = tfidfvector.fit(x_variableTrain2)
x_train2_tfidf_matrix = tfidfvector.transform(x_variableTrain2)
x_test2_tfidf_matrix = tfidfvector.transform(x_variableTest2)

multi_nb = nb.MultinomialNB(alpha=1,fit_prior=True)
mb = multi_nb.fit(x_train2_tfidf_matrix, y_variableTrain2)
multi_nb_pred2 = mb.predict(x_test2_tfidf_matrix)

accuracy_dataset2 = metrics.accuracy_score(y_variableTest2, multi_nb_pred2)
accuracy_dataset2*=100
print("Accuray on Dataset2 (Airline Data) Using Naive Bayes: %.2f" %(accuracy_dataset2) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest2, multi_nb_pred2),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest2, multi_nb_pred2, target_names=categories_df2))

end = time.time()
print ("Runtime is ", (end-start), "seconds")

"""To check number of elements in testing data and predicted data
import collections
collections.Counter(y_variableTest2)
collections.Counter(multi_nb_pred2)
"""

"""
Function to create visual representation of confusion matrix: 
Using http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, categories_df2, rotation=45)
    plt.yticks(tick_marks, categories_df2)
    plt.tight_layout()
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    
#Confusion matrix for Debate Data
cm = conmat(y_variableTest1, multi_nb_pred1)
np.set_printoptions(precision=2)
print('Confusion matrix for Debate Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

#Confusion matrix for Airline Data
cm = conmat(y_variableTest2, multi_nb_pred2)
np.set_printoptions(precision=2)
print('Confusion matrix for Airline Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

#Multinomial Log1istics regression
start = time.time()

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.5, sublinear_tf = True,stop_words = 'english')

#Model for Dataset 1
x_train1_tfidf_matrix = tfidfvector.fit(x_variableTrain1)
x_train1_tfidf_matrix = tfidfvector.transform(x_variableTrain1)
x_test1_tfidf_matrix = tfidfvector.transform(x_variableTest1)

log_reg = lm.LogisticRegression(multi_class='multinomial',solver='newton-cg',C=25,random_state=1)
lr = log_reg.fit(x_train1_tfidf_matrix, y_variableTrain1)
log_reg_pred1 = lr.predict(x_test1_tfidf_matrix)

accuracy_dataset1 = metrics.accuracy_score(y_variableTest1, log_reg_pred1)
accuracy_dataset1*=100
print("Accuray on Dataset1 (Debate Data) Using Logistics Regression: %.2f" %(accuracy_dataset1) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest1, log_reg_pred1),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest1, log_reg_pred1, target_names=categories_df1))

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.5, sublinear_tf = True,stop_words = 'english')

#Model for Dataset 2
x_train2_tfidf_matrix = tfidfvector.fit(x_variableTrain2)
x_train2_tfidf_matrix = tfidfvector.transform(x_variableTrain2)
x_test2_tfidf_matrix = tfidfvector.transform(x_variableTest2)

log_reg = lm.LogisticRegression(multi_class='multinomial',solver='newton-cg',C=25,random_state=1)
lr = log_reg.fit(x_train2_tfidf_matrix, y_variableTrain2)
log_reg_pred2 = lr.predict(x_test2_tfidf_matrix)

accuracy_dataset2 = metrics.accuracy_score(y_variableTest2, log_reg_pred2)
accuracy_dataset2*=100
print("Accuray on Dataset2 (Airline Data) Using Logistics Regression: %.2f" %(accuracy_dataset2) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest2, log_reg_pred2),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest2, log_reg_pred2, target_names=categories_df2))

end = time.time()
print ("Runtime is ", (end-start), "seconds")

#Confusion matrix for Debate Data
cm = conmat(y_variableTest1, log_reg_pred1)
np.set_printoptions(precision=2)
print('Confusion matrix for Debate Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Confusion matrix for Airline Data
cm = conmat(y_variableTest2, log_reg_pred2)
np.set_printoptions(precision=2)
print('Confusion matrix for Airline Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

#k-Nearest Neighbours
start = time.time()

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.5, sublinear_tf = True,stop_words = 'english')

#Model for Dataset 1
x_train1_tfidf_matrix = tfidfvector.fit(x_variableTrain1)
x_train1_tfidf_matrix = tfidfvector.transform(x_variableTrain1)
x_test1_tfidf_matrix = tfidfvector.transform(x_variableTest1)

k_nearest = kNN(n_neighbors=12,metric="minkowski",random_state=1)
k_n_n = k_nearest.fit(x_train1_tfidf_matrix, y_variableTrain1)
knn_pred1 = k_n_n.predict(x_test1_tfidf_matrix)
knn_pred1_train = k_n_n.predict(x_train1_tfidf_matrix)

accuracy_dataset1 = metrics.accuracy_score(y_variableTest1, knn_pred1)
accuracy_dataset1_train = metrics.accuracy_score(y_variableTrain1, knn_pred1_train)
accuracy_dataset1*=100
accuracy_dataset1_train*=100
print("Accuray on Testing Dataset1 (Debate Data) Using k-Nearest Neighbour: %.2f" %(accuracy_dataset1) + "%")
print("Accuray on Training Dataset1 (Debate Data) Using k-Nearest Neighbour: %.2f" %(accuracy_dataset1_train) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest1,knn_pred1),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest1, knn_pred1, target_names=categories_df1))

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.5, sublinear_tf = True,stop_words = 'english')
        
#Model for Dataset 2
x_train2_tfidf_matrix = tfidfvector.fit(x_variableTrain2)
x_train2_tfidf_matrix = tfidfvector.transform(x_variableTrain2)
x_test2_tfidf_matrix = tfidfvector.transform(x_variableTest2)

k_nearest = kNN(n_neighbors=7,metric="minkowski",random_state=1)
k_n_n = k_nearest.fit(x_train2_tfidf_matrix, y_variableTrain2)
knn_pred2 = k_n_n.predict(x_test2_tfidf_matrix)
knn_pred2_train = k_n_n.predict(x_train2_tfidf_matrix)

accuracy_dataset2 = metrics.accuracy_score(y_variableTest2, knn_pred2)
accuracy_dataset2_train = metrics.accuracy_score(y_variableTrain2, knn_pred2_train)
accuracy_dataset2*=100
accuracy_dataset2_train*=100
print("Accuray on Testing Dataset2 (Airline Data) Using k-Nearest Neighbour: %.2f" %(accuracy_dataset2) + "%")
print("Accuray on Training Dataset2 (Airline Data) Using k-Nearest Neighbour: %.2f" %(accuracy_dataset2_train) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest2, knn_pred2),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest2, knn_pred2, target_names=categories_df2))

end = time.time()
print ("Runtime is ", (end-start), "seconds")

#Confusion matrix for Debate Data
cm = conmat(y_variableTest1, knn_pred1)
np.set_printoptions(precision=2)
print('Confusion matrix for Debate Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Confusion matrix for Airline Data
cm = conmat(y_variableTest2, knn_pred2)
np.set_printoptions(precision=2)
print('Confusion matrix for Airline Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

#Random forest classifier
start = time.time()

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.5, sublinear_tf = True,stop_words = 'english')

#Model for Dataset 1
x_train1_tfidf_matrix = tfidfvector.fit(x_variableTrain1)
x_train1_tfidf_matrix = tfidfvector.transform(x_variableTrain1)
x_test1_tfidf_matrix = tfidfvector.transform(x_variableTest1)

randomforest = rfc(n_estimators=40, random_state=1)
r_f_c = randomforest.fit(x_train1_tfidf_matrix, y_variableTrain1)
rfc_pred1 = r_f_c.predict(x_test1_tfidf_matrix)

accuracy_dataset1 = metrics.accuracy_score(y_variableTest1, rfc_pred1)
accuracy_dataset1*=100
print("Accuray on Dataset1 (Debate Data) Using Random Forest: %.2f" %(accuracy_dataset1) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest1, rfc_pred1),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest1, rfc_pred1, target_names=categories_df1))

#To create TFxIDF Matrix by converting document terms into a TF-IDF matrix
tfidfvector = TfIdf(min_df = 0.01, max_df = 0.5, sublinear_tf = True,stop_words = 'english')

#Model for Dataset 2
x_train2_tfidf_matrix = tfidfvector.fit(x_variableTrain2)
x_train2_tfidf_matrix = tfidfvector.transform(x_variableTrain2)
x_test2_tfidf_matrix = tfidfvector.transform(x_variableTest2)

randomforest = rfc(n_estimators=40, random_state=1)
r_f_c = randomforest.fit(x_train2_tfidf_matrix, y_variableTrain2)
rfc_pred2 = r_f_c.predict(x_test2_tfidf_matrix)

accuracy_dataset2 = metrics.accuracy_score(y_variableTest2, rfc_pred2)
accuracy_dataset2*=100
print("Accuray on Dataset2 (Airline Data) Using Random Forest: %.2f" %(accuracy_dataset2) + "%")
print("Confusion Matrix:\n",conmat(y_variableTest2, rfc_pred2),"\n")
print("Classification Report:\n",metrics.classification_report(y_variableTest2, rfc_pred2, target_names=categories_df2))

end = time.time()
print ("Runtime is ", (end-start), "seconds")

#Confusion matrix for Debate Data
cm = conmat(y_variableTest1, rfc_pred1)
np.set_printoptions(precision=2)
print('Confusion matrix for Debate Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Confusion matrix for Airline Data
cm = conmat(y_variableTest2, rfc_pred2)
np.set_printoptions(precision=2)
print('Confusion matrix for Airline Data')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
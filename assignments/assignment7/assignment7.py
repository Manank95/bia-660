
# coding: utf-8

# In[25]:

# Reference from Lecture notes of clustering, Topic modelling and classification for (Task3)
#assignment 7
# Task 1 - K-mean clustering.
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.cluster import KMeansClusterer, cosine_distance
import numpy as np
import pandas as pd
# for task 2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report


def task1():
    # initialize the TfidfVectorizer 
    # set min document frequency to 5
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=5)
    dtm= tfidf_vect.fit_transform(text)

    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=10)
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)
    print("Task 1: (a)-----------")
    print(clusters[0:5])

    # get the centroids, means to what these clusters are about and we can give meaningful names
    centroids=np.array(clusterer.means())
    # sort in reverese order and get feature names
    sorted_centroids = centroids.argsort()[:, ::-1]
    voc_lookup= tfidf_vect.get_feature_names()

    print("\nTo get top 20 words from each cluster.")
    for i in range(num_clusters):
        # get words with top 20 tf-idf weight in the centroid
        top_words=[voc_lookup[word_index] for word_index in sorted_centroids[i, :20]]
        print("Cluster %d: %s " % (i, "; ".join(top_words)))

    print("\n")
    #external evaluation/ majority vote rule
    df=pd.DataFrame(list(zip(first_label, clusters)), columns=['actual_class','cluster'])
    print("Task 1: (b)-----------\n")
    print(df.head())
    print(pd.crosstab( index=df.cluster, columns=df.actual_class))

    cluster_dict={0:'T1', 1:'T2', 2:'T3'}
    # Assign true class to cluster
    predicted_target=[cluster_dict[i] for i in clusters]
    #predicted_target
    print("\n\nTask 1: (c) ----------\n")
    print(metrics.classification_report(first_label, predicted_target))
    print("\n\nTask 1: (d) ----------\n")
    print("meaningful names to each cluster based on cluster centroids/samples.")
    print("Cluster 0: com.accidents (T1)\nCluster 1: com.disasters (T2)\nCluster 2: com.economy.report (T3)")
    
    

def task2():
    #Task-2 LDA (single label)
    tf_vectorizer = CountVectorizer(max_df=0.90, min_df=50, stop_words='english')
    tf = tf_vectorizer.fit_transform(text)
    tf_feature_names = tf_vectorizer.get_feature_names()
    X_train, X_test = train_test_split(tf, test_size=0.1, random_state=0)
    num_topics = 3
    num_top_words=20
    lda = LatentDirichletAllocation(n_components=num_topics,                                     max_iter=10,verbose=1,
                                    evaluate_every=1, n_jobs=1,
                                    random_state=0).fit(X_train)
    
    for topic_idx, topic in enumerate(lda.components_):
        print ("Topic %d:" % (topic_idx))
        # print out top 10 words per topic 
        words=[(tf_feature_names[i],topic[i]) for i in topic.argsort()[::-1][0:num_top_words]]
        print(words)
        print("\n")
    
    topics_assign=lda.transform(X_train)
    print(topics_assign[0:5])
    print("\n\n")
    #topics_assign = topics_assign[~np.all(topics_assign == 0, axis=1)]
    #print(topics_assign[0:5])

    topics_assign_list = np.argmax(topics_assign, axis=1).tolist()
    print(topics_assign_list[0:5])
    print("\n\n")
    
    
    df=pd.DataFrame(list(zip(first_label, topics_assign_list)), columns=['actual_class','topics'])
    print(df.head())
    print(pd.crosstab( index=df.topics, columns=df.actual_class))
    
    cluster_dict={0:'T1', 1:'T2', 2:'T3'}
    # Assign true class to cluster
    predicted_target=[cluster_dict[i] for i in topics_assign_list]
    #print(len(predicted_target))
    #print(len(all_labels))
    print("\n\nTask 2: (c) ----------\n")
    print(metrics.classification_report(first_label[0:3618], predicted_target))
    print("\n\nTask 2: (d) ----------\n")
    print("meaningful names to each cluster based on cluster centroids/samples.")
    print("Cluster 0: com.oil.news (T1)\nCluster 1: com.accidents (T2)\nCluster 2: com.economy.budget (T3)")
    
    
    # Task 3-----------
    
    
    print("\n\nTask 3 (1) - applying threshold")
    prob_threshold=0.4
    topics=np.copy(topics_assign)
    topics=np.where(topics>=prob_threshold, 1, 0)
    print(topics[0:5])
    
    #multilabel classification
    print("\n\nTask 3 (2) - Applying multilabel classification for all-labels")
    mlb = MultiLabelBinarizer()
    Y=mlb.fit_transform(all_labels)
    X_train, X_test, Y_train, Y_test = train_test_split(text, Y, test_size=0.3, random_state=0)
    classifier = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words="english", min_df=2)),
    ('clf', OneVsRestClassifier(LinearSVC()))])

    classifier.fit(X_train, Y_train)
    predicted = classifier.predict(X_test)
    print(classification_report(Y_test, predicted, target_names=mlb.classes_))
    
    print(precision_score(Y_test, predicted, average="macro"))
    print(recall_score(Y_test, predicted, average="macro"))
    print(f1_score(Y_test, predicted, average="macro"))

    
    #print(metrics.classification_report(all_labels, predicted_target))

    # Task 3 (3) determine the best threshold
    print("\n\nDetermine the best threshold from 0 to 1")
    tst_thr=0
    while tst_thr<=1:
        topics=np.copy(topics_assign)
        topics=np.where(topics>=tst_thr, 1, 0)
        print("Threshold = ",tst_thr)
        print(topics[0:5])
        print("\n")
        tst_thr+=.05
    
    
if __name__ == "__main__":
    data=json.load(open('ydata_3group.json','r'))
    num_clusters=3

    text,first_label,all_labels=zip(*data)
    text=list(text)
    first_label=list(first_label)
    all_labels = list(all_labels)
    #rint(text[0])
    #rint(first_label[0])
    task1()
    print("\n\n")
    task2()
    # Task 3 is included in the last function call.


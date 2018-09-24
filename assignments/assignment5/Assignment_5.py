
# coding: utf-8

# # Assignment 5: Natural Language Processing - Collocations and TF-IDF 

# ## 1.  collocations
# - Define a function top_collocation(tokens, K) to find top-K collocations in specific patterns in a document as follows:
#   - takes a list of tokens and K as inputs
#   - uses the following steps to find collocations:
#     - POS tag each token
#     - create bigrams
#     - get frequency of each bigram (you can use nltk.FreqDist)
#     - keep only bigrams matching the following patterns:
#        - Adj + Noun: e.g. linear function
#        - Noun + Noun: e.g. regression coefficient
#   - returns top K collocations by frequency

# ## 2. Document search by TF-IDF
# 
# 1. Modify tfidf and get_doc_tokens functions in Section 7.5 of your lecture notes to add “normalize” as a parameter. This parameter can take two possible values: None, "stem". The default value is None; if this parameter is set to "stem", stem each token. 
# 2. In the main block, do the following:
#     1. Read the dataset “amazon_review_300.csv”. This dataset has 3 columns: label, title, review. We’ll use “review” column only in this assignment.
#     2. Calculate the tf-idf matrix for all the reviews using the modified functions tfidf function, each time with a different “normalize” value 
#     3. Take any review from your dataset, for each "normalize" option, find the top 5 documents most similar to the selected review, and print out these reviews
#     4. Check if the top 5 reviews change under different "normalize" options. Which option do you think works better for the search? Write down your analysis as a print-out, or attach a txt file if you wish.
#     5. (**bouns**) For each pair of similar reviews you find in (C), e.g. review x is similar to review y, find matched words under each "normalize" option. Print out top 10 words contributing most to their cosine similarity. (Hint: you need to modify the tfidf function to return the set of words as a vocabulary)

# In[65]:


# add import statement
#reference - lecture notes NLP2
import nltk, re, string
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
# numpy is the package for matrix cacluation
import numpy as np  
import pandas as pd
from scipy.spatial import distance
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

stop_words = stopwords.words('english')

def top_collocation(tokens, K):
    result=[]
    
    # add your code here
    tagged_tokens= nltk.pos_tag(tokens)
    #print(tagged_tokens)
    bigrams=nltk.bigrams(tagged_tokens)
    #bigram_dist=nltk.FreqDist(bigrams)
    #print(bigram_dist)
    phrases=[ (x[0],y[0]) for (x,y) in bigrams          if (x[1].startswith('JJ') or x[1].startswith('NN'))         and y[1].startswith('NN')]
    #print(phrases)
    
    bigram_dist=nltk.FreqDist(phrases)
    result = bigram_dist.most_common(K)
    
    return result


# modify these two functions
def get_doc_tokens(doc, normalize):
       
    # you can add bigrams, collocations, stemming, 
    # or lemmatization here
    if normalize == "stem":
        tokens=[porter_stemmer.stem(token.strip())             for token in nltk.word_tokenize(doc.lower())             if token.strip() not in stop_words and               token.strip() not in string.punctuation]
    else:
        tokens=[token.strip()             for token in nltk.word_tokenize(doc.lower())             if token.strip() not in stop_words and               token.strip() not in string.punctuation]
        
    token_count={token:tokens.count(token) for token in set(tokens)}
    
    #print(tokens)
    #print(token_count)
    return token_count

def tfidf(docs, normalize):
    docs_tokens={idx:get_doc_tokens(doc, normalize)              for idx,doc in enumerate(docs)}
    
    #print(docs_tokens)
    print("\n------------\n")
    # step 3. get document-term matrix
    dtm=pd.DataFrame.from_dict(docs_tokens, orient="index" )
    dtm=dtm.fillna(0)
      
    # step 4. get normalized term frequency (tf) matrix        
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf.T, doc_len).T
    
    # step 5. get idf
    df=np.where(tf>0,1,0)
    #idf=np.log(np.divide(len(docs), \
    #    np.sum(df, axis=0)))+1

    smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1 
    # apply normalize(tf*smoothed_idf) for noralized values
    smoothed_tf_idf=tf*smoothed_idf
    #smoothed_tf_idf=tf*smoothed_idf
    
    return smoothed_tf_idf

            


# In[70]:


import nltk
import csv

if __name__ == "__main__":  
    
    # test collocation
    text=nltk.corpus.reuters.raw('test/14826')
    tokens=nltk.word_tokenize(text.lower())
    #print(tokens)
    print("\n------------\n")
    print(top_collocation(tokens, 10))
    print("\n------------\n")
    
    # load data
    docs=[]
    with open("amazon_review_300.csv","r") as f:
        reader=csv.reader(f)
        for line in reader:
            docs.append(line[2])
    
    # Find similar documents -- No STEMMING
    tf_idf = tfidf(docs, None)
    print(tf_idf)
    
    # find top 5 docs similar to first one -- NO STEMMING
    similarity=1-distance.squareform    (distance.pdist(tf_idf, 'cosine'))
    
    res1 = np.argsort(similarity)[:,::-1][0,0:6]
    print(res1)
            
            
    # Find similar documents -- STEMMING
    stem_tf_idf = tfidf(docs, "stem")
    print(stem_tf_idf)

    # find top 5 docs similar to first one -- STEMMING
    print("\n------------\n")
    similarity=1-distance.squareform    (distance.pdist(stem_tf_idf, 'cosine'))
    
    res = np.argsort(similarity)[:,::-1][0,0:6]
    print(res)
    print("\n Finding similar docs without stemming gives better and accurate results as compared to with stemming. By comparing both results we can say that results and top lines without stemming gives matches which is more similar to the input line.")
    print("\n bonus")
    d1 = similarity[0].tolist()
    print(sorted(enumerate(d1), key=lambda item:-item[1])[0:9])
    
    


# Manank Valand - 10429101
# references: lecture notes - Natural Language Processing I.ipynb, Regular_Expression.ipynb and Python_II.ipynb (for csv)
#
# wait for 2-3 seconds to get full output, because it takes time to read and apply calculations on 
# whole bunch of texts for couple of times
#
# PUT all required files for this assignment in the same path of this program. 
# Or change path values in driver program (__main__) 


import string
import nltk
import re
import csv
from nltk.corpus import stopwords

def tokenize(text):
    tokens=[]
    # write your code here
    stop_words = stopwords.words('english')
    text = text.lower()
    #print(text)
    #pattern = r'\w'
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if re.match("^[A-Za-z_-]*$", token)]
    #print(tokens)
    # to remove punctuations from begging and starting of the tokens 
    tokens = [token.strip(string.punctuation) for token in tokens]
    # now removing extra empty characters from tokens
    tokens = [token.strip() for token in tokens if token.strip()!='']
    tokens = [token for token in tokens if len(token)>1]
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def sentiment_analysis(text, positive_words, negative_words):
    
    sentiment=None
    negations=['not', 'too', 'n\'t', 'no', 'cannot', 'neither', 'nor']
    # write your code here
    tokens = tokenize(text)
    #print(tokens)
    positive_tokens =[]
    negative_tokens =[]
    for idx, token in enumerate(tokens):
        if token in positive_words:
            if(idx>0):
                if tokens[idx-1] not in negations:
                    positive_tokens.append(token)
                else:
                    negative_tokens.append(token)
            else:
                positive_tokens.append(token)
        elif token in negative_words:
            if(idx>0):
                if tokens[idx-1] not in negations:
                    negative_tokens.append(token)
                else:
                    positive_tokens.append(token)
            else:
                negative_tokens.append(token)
    #remove below 2 comments to check the array built out of provided string
    #print("positive tokens:",positive_tokens)
    #print("negative tokens:",negative_tokens)
    
    if len(positive_tokens)>len(negative_tokens):
        sentiment = 2
    else:
        sentiment = 1
    return sentiment


def performance_evaluate(input_file, positive_words, negative_words):
    
    accuracy=None
    cnt=0
    # write your code here
    with open(input_file, "r") as f:
        reader=csv.reader(f, delimiter=',')
        rows=[(row[0], row[2]) for row in reader]
    row_len = len(rows)
    #print(row_len)
    for i in rows:
        if int(i[0]) == sentiment_analysis(i[1], positive_words, negative_words):
            cnt+=1
    
    #print(cnt)
    accuracy = cnt/row_len
    return accuracy


# In[23]:


if __name__ == "__main__":  
    
    text="This is a breath-taking ambitious movie; test text: abc_dcd abc_ dvr89w, abc-dcd -abc"

    tokens=tokenize(text)
    print("tokens:")
    print(tokens)
    
    
    with open("positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
        
    with open("negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
    #print(positive_words)  
    print("\nsentiment")
    sentiment=sentiment_analysis(text, positive_words, negative_words)
    print(sentiment)
    
    accuracy=performance_evaluate("amazon_review_300.csv", positive_words, negative_words)
    print("\naccuracy")
    print(accuracy)


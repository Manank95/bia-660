
# coding: utf-8
# Manank Valand - 10429101 - reference taken from lecture notes

# Structure of your solution to Assignment 1
import numpy as np
import pandas as pd

def analyze_tf(arr):
    
    A = np.sum(arr, axis=1)
    
    #print(arr)
    #print(A)
    tf = arr.T / A
    #print(tf.T)
    x=np.where(tf.T>0,1,0)
    #print(x)
    df = np.sum(x, axis=0)
    #print(df)
    D = tf.T / df
    #print(D)
    tf_idf=np.argsort(D)
    #print(tf_idf)
    #print(tf_idf[:,::-1][:,0:3])
    
    return tf_idf[:,::-1][:,0:3]

def analyze_cars():
    
    # add your code
    data=pd.read_csv('cars.csv', header=0)
    print(data)
    sort = data.sort_values(by=['cylinders','mpg'], ascending=False)
    print(sort.head(3))
    sort['brand'] = sort.apply(lambda row:         row["car"].split()[0], axis=1)
    print(sort)
    tmp = sort[(sort.brand=='ford') | (sort.brand=='buick') | (sort.brand=='honda')]
    #print(tmp)
    grouped=tmp.groupby('cylinders')
    print(grouped.mean())
    print(grouped.min())
    print(grouped.max())
    print(pd.crosstab(index=sort.brand, columns=sort.cylinders, values=sort.mpg, aggfunc=np.average ))
    

# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    #1 Test Question 1
    arr=np.random.randint(0,3,(4,8))
    #print(arr)
    tf_idf=analyze_tf(arr)
    print(tf_idf)
    
    # Test Question 2
    analyze_cars()


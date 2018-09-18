
# coding: utf-8

import numpy as np
import csv

def count_token(text):
    
    token_count={}
    
    # add your code
    arr = text.split()
    for i in arr:
        i.rstrip()
        i.lstrip()
        if len(i)<=1:
            arr.remove(i)
    
    for j in range(len(arr)):
        arr[j] = arr[j].lower()
    for k in arr:
        if k in token_count.keys():
            token_count[k]+=1
        else:
            token_count[k]=1
    return token_count
    #return token_count

class Text_Analyzer(object):
    
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        # add your code
          
    def analyze(self):
        f = open(self.input_file, "r")
        concat_string = ""
        for x in f:
            concat_string+=x
        f.close()
        #print(concat_string)
        token_dict = count_token(concat_string)
        #print(token_dict)
        with open(self.output_file, "w") as f:
            writer=csv.writer(f, delimiter=',')
            for key, value in token_dict.items():
                writer.writerow([key,value])
        

def analyze_tf(arr):
    
    tf_idf=None
    
    # add your code
    
    return tf_idf

# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    # Test Question 1
    text='''Hello world!
        This is a hello world example !'''   
    print(count_token(text))
    
    # # The output of your text should be: 
    # {'this': 1, 'is': 1, 'example': 1, 'world!': 1, 'world': 1, 'hello': 2}
    
    # Test Question 2
    analyzer=Text_Analyzer("foo.txt", "foo.csv")
    vocabulary=analyzer.analyze()
    
    #3 Test Question 3
    arr=np.random.randint(0,3,(4,8))

    tf_idf=analyze_tf(arr)


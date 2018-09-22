
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

def mpg_plot():
    # put your code here
    df = pd.read_csv('auto-mpg.csv', header=0)
    #print(df.head())
    df1 = df.loc[df['origin']==1]
    df2 = df.loc[df['origin']==2]
    df3 = df.loc[df['origin']==3]
    #print(df1)
    df1.groupby('model_year')["mpg"].mean()    .plot(kind='line', figsize=(8,4), label = '1')    .legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
    df2.groupby('model_year')["mpg"].mean()    .plot(kind='line', figsize=(8,4), label = '2')    .legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
    df3.groupby('model_year')["mpg"].mean()    .plot(kind='line', figsize=(8,4), label = '3')    .legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
    plt.show()
    
def getReviews(movie_id):
    reviews=[] # variable to hold all reviews
    page_url="https://www.rottentomatoes.com/m/"+movie_id+"/reviews/?type=top_critics"
    page = requests.get(page_url)
    if page.status_code==200:
        # insert your code to process page content
        soup = BeautifulSoup(page.content, 'html.parser')
        divs=soup.select("div#reviews div.content div.review_table div.row")
        #print(len(divs))
        
        for idx, div in enumerate(divs):
            reviewer = None
            date = None
            description = None
            score = None
            
            p_reviewer = div.select("div div.critic_name a.unstyled")
            #print(p_reviewer)
            if p_reviewer!=[]:
                reviewer=p_reviewer[0].get_text()
            
            p_date = div.select("div.review_container div.review_area div.review_date")
            #print(p_date)
            if p_date!=[]:
                date = p_date[0].get_text()
                
            p_description = div.select("div.review_container div.review_area div.review_desc div.the_review")
            #print(p_description)
            if p_description!=[]:
                description = p_description[0].get_text()
                
            p_score = div.select("div.review_container div.review_area div.review_desc div.small")
            if p_score!=[]:
                tmp = p_score[0].get_text().split()[-1]
                if tmp =="Review":
                    score = 'not available'
                else:
                    score = p_score[0].get_text().split()[-1]
            reviews.append((reviewer, date, description, score))
            
    return reviews

if __name__ == "__main__":
    mpg_plot()
    movie_id='black_panther_2018'
    reviews=getReviews(movie_id)
    print(reviews)


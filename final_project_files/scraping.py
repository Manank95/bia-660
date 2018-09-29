import requests
from bs4 import BeautifulSoup
import pandas as pd


filename = "9D.csv"


# change the count, it will indicate the page number. In each page we have 50 results.
cnt = 1050

# change this link for every new result

link = "?q=data+scientist&l=&cb=jt&start="+str(cnt)

df1 = pd.DataFrame()

for i in range(0,4):

    df = []
    # send a get request to the web page
    rows = []
    next_link = []
    page_url = "https://www.indeed.com/resumes" + link
    page = requests.get(page_url)


    # send a get request to the web page

    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')

        #print(soup.prettify())

        # for a in soup.find_all("a", class_='next'):
        #     next_link.append(a["href"])

        print("\n\n")
        # link = next_link[0]


        for a in soup.find_all("a", class_='app_link'):
            rows.append(a["href"])
            print(a["href"])
        #print(rows[2])
        print(cnt)
        cnt = cnt+len(rows)

        title_lists = []
        company_lists = []
        date_lists = []
        description_lists = []
        major_lists = []
        school_lists = []
        skills_list = []
        eligibility_list = []
        summary_list = []
        headline_list = []



        for j in range(0, len(rows)-1):

            title_list = []
            company_list = []
            date_list = []
            description_list = []
            major_list = []
            school_list = []



            res_page = requests.get("https://www.indeed.com" + rows[j])
            if res_page.status_code == 200:
                res_soup = BeautifulSoup(res_page.content, 'html.parser')
                resume = []
                headline = None
                summary = None
                eligibility = None
                skills = None

                # get headine of resume
                res_headline = res_soup.select("h2#headline")
                if res_headline != []:
                    headline = res_headline[0].get_text()
                    headline_list.append(headline)
                else:
                    headline_list.append("NA")

                # get summary of resume
                res_summary = res_soup.select("p#res_summary")
                if res_summary != []:
                    summary = res_summary[0].get_text()
                    summary_list.append(summary)
                else:
                    summary_list.append("NA")

                # get work authorization or work eligibility
                res_eligibility = res_soup.select("p#employment_eligibility")
                if res_eligibility != []:
                    eligibility = res_eligibility[0].get_text()
                    eligibility_list.append(eligibility)
                else:
                    eligibility_list.append("NA")

                # get skills of resume
                res_skills = res_soup.select("div#skills-items")
                if res_skills != []:
                    skills = res_skills[0].get_text()
                    skills_list.append(skills)
                else:
                    skills_list.append("NA")


                workex_divs = res_soup.select("div#work-experience-items div.data_display")

                work_exp_list = []

                for idx, div in enumerate(workex_divs):
                    w_title = None
                    w_company = None
                    w_dates = None
                    w_description = None

                    # get work title
                    p_title = div.select("p.work_title")
                    if p_title != []:
                        w_title = p_title[0].get_text()
                        title_list.append(w_title)
                    else:
                        title_list.append("NA")

                    p_company = div.select("div.work_company")
                    if p_company != []:
                        w_company = p_company[0].get_text()
                        company_list.append(w_company)
                    else:
                        company_list.append("NA")

                    p_dates = div.select("p.work_dates")
                    if p_dates != []:
                        w_dates = p_dates[0].get_text()
                        date_list.append(w_dates)
                    else:
                        date_list.append("NA")

                    p_description = div.select("p.work_description")
                    if p_description != []:
                        w_description = p_description[0].get_text()
                        description_list.append(w_description)
                    else:
                        description_list.append("NA")

                        # .replace(u'\xa0', u' ') to remove \xa0's from everywhere - and then just filter them out


                education_divs = res_soup.select("div#education-items div.education-section")
                if education_divs != []:
                    for idx, div in enumerate(education_divs):
                        e_title = None
                        e_school = None

                        # get educaiton title
                        t_title = div.select("p.edu_title")
                        if t_title != []:
                            e_title = t_title[0].get_text()
                            major_list.append(e_title)
                        else:
                            major_list.append("NA")

                        t_school = div.select("div.edu_school")

                        if t_school != []:
                            e_school = t_school[0].get_text()
                            school_list.append(e_school)
                        else:
                            school_list.append("NA")

                # educations_list and education_list both are different

            title_lists.append(title_list)
            company_lists.append(company_list)
            date_lists.append(date_list)
            description_lists.append(description_list)
            major_lists.append(major_list)
            school_lists.append(school_list)




        raw_data = {'headline': headline_list,
                    'summary': summary_list,
                    'eligibility': eligibility_list,
                    'skills': skills_list,
                    'job titles': title_lists,
                    'company name' : company_lists,
                    'dates' : date_lists,
                    'description':description_lists,
                    'school':school_lists,
                    'majors':major_lists}

        #print(raw_data)

        df = pd.DataFrame(raw_data, columns=['headline', 'summary', 'eligibility', 'skills','job titles', 'company name','dates','description','school','majors'])



        with open(filename, 'a') as f:
            df.to_csv(f, header=False)
            f.close()

    # performance_evaluate("indeed_scraped_data_science.csv")
    # dateToSum("indeed_scraped_data_science.csv")
    # description follwing qualities
    # skills - pig
    # autrize not authorize
    # summary
    # education - bachleor,masters 1 gram and 2 gram
    # preprocessin.py
    # analysis.py
    # .csv
    # readme about project
    # next plan .txt


# from sklearn.feature_extraction.text import TfidfVectorizer
# import csv
# import pandas as pd
#
#
# def performance_evaluate(filename):
#     with open(filename, "r", encoding="ISO-8859-1") as f:
#         reader = csv.reader(f, delimiter=',')
#         text = [(row[1]) for row in reader]
#
#     tfidf_vect = TfidfVectorizer(input=text, encoding='utf-8', decode_error='replace', stop_words="english",
#                                  ngram_range=(1, 3))
#     dtm = tfidf_vect.fit_transform(text)
#     print("type of dtm:", type(dtm))
#     print("size of tfidf matrix:", dtm.shape)
#     voc_lookup = {tfidf_vect.vocabulary_[word]: word \
#                   for word in tfidf_vect.vocabulary_}
#
#     doc0 = dtm[0].toarray()[0]
#     print(doc0.shape)
#
#     # get index of top 20 words
#     top_words = (doc0.argsort())[::-1][0:20]
#     print([(voc_lookup[i], doc0[i]) for i in top_words])
#
#
# if __name__ == "__main__":
#     performance_evaluate("indeed_scraped_data_science.csv")



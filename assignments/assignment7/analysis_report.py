1) Based on cluster centroids/samples, give a meaningful name (instead of T1, T2, T3) to each

T1 -> com.accidents
T2 -> com.disasters
T3 -> com.economy.report

# ################## Task 2 #########################

2) Calculate precision/recall/f-score and compare them with the results in Task 1.
Task 1 performs better than Task 2. Classification_report of Task2 is lower than Task 1.

2) Based on word probabilities in each topic, give the topic a meaningful name.
T1 -> com.oil.news
T2 -> com.accidents
T3 -> com.economy.budget

# ################## Task 3 #########################

2) Write your analysis about the results, and conclude what can be the best threshold value.
With increase in threshold value precision also increases and recall decreases.

All classification_reports are constant initially and they gradually increase from 0.8
The Best threshold value is between 0.45, then 0.90 and lastly 0.45.
# COS427TermProjectDominicPollettaJosephPolletta
Term Project for USM Fall 2021 Computational Text Analytics for Dominic Polletta and Joseph Polletta

Team Members: Dominic Polletta and Joseph Polletta

Objective for the project: The objective is to classify abstract texts of medical publications obtained from the PubMed database to test the different machine learning models on real world data. We will obtain a large amount of journal articles from PubMed, extract the abstracts from each article to be used as a Train/Test set for Naive Bayes, Support Vector Machine, and Logistic Regression machine learning models. Using the same test data, we will inspect the different qualities of each model, evaluating their benefits and downsides, both on their own and when compared to other machine learning models used.

Materials: We will be utilizing a set of abstracts pulled from articles we will obtain from the PubMed database for four different diseases, Acute Rheumatic Arthritis, Cardiovascular Abnormalities, Lyme Disease, and Knee Osteoarthritis.

Methods: We will be normalizing the abstracts, then converting them into a standardized Pandas DataFrame, which we will then use as the input for three different types of machine learning models. The models we will be using are Naive Bayes, Support Vector Machine (SVM), Logistic Regression.

Experimental Validation: 

Results: 

Conclusion: We found that Logistic Regression was the most accurate machine learning model for our dataset, but also had the longest run time by far. At this scale of dataset, the increased time cost is managable but at a larger dataset it is possible that the longer time to run may not be worth it compared to another model.

Discussion: One aspect of the project that we could attempt in the future to improve accuracy would involve the way that we break down the data for analysis. In this project we utilized a homemade Bag of Words algorithm to serve as the data input to our machine learning models. However, Bag of Words does not take into account the order of the words. If we were to use a different method, such as Word2Vec, which takes into account the order of the words, then it may be possible to further improve our results. Another worry that occured when accumulating the data was the unbalanced nature of the classes. At the start, we were worried that the fact that two of our classes were at 10,000 examples while our other two close to a 1/4 of that number would impact the accuracy significantly. We were prepared to manually adjust weighting of the data to ensure that the classes were handled in a balanced manner. However, we did not see any signs of this being an issue with our results. We hypothesize that the SKLearn package was able to hangle the inbalanced classes, as there were enough in each to ensure that there wasn't issues with too small a dataset.

Outlook: We now have a baseline reading for different machine learning models on medical journal abstracts. There several avenues for expanding our efforts, such as attempting the same tests but with different types of texts such as social media posts. Another avenue would be to use Word2Vec instead of Bag Of Words, or to attempt using more types of machine learning models to further evaluate.

# Joseph Polletta & Dominic Polletta
# Term Project

# Import regex, nltk, numpy, pandas, and sklearn
import regex
import nltk
import numpy as np
import pandas as pd
import sklearn

# Download and import necessary features from packages
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define stopword list, tokenizer, stemmer, lemmatizer, and NBmodel
StopWordList = nltk.corpus.stopwords.words('english')
Tokenizer = TreebankWordTokenizer()
WordStemmer = PorterStemmer()
Lemmatizer = WordNetLemmatizer()
NBModel = MultinomialNB()

# Define a method to normalize the corpus and generate a bag of words representation for it
def NormalizeAndBOW(InputToNormalize):
    # Turn all text in InputToNormalize into lowercase
    LowercaseText = InputToNormalize.lower()

    # Remove punctuation from LowercaseText
    LowercaseTextNoPunct = regex.sub("[=%\.\?!,:;}{\)\(\]\['\"-\/]", " ", LowercaseText)

    # Gather tokens
    StopWordTokens = Tokenizer.tokenize(LowercaseTextNoPunct)

    # Eliminate stop words
    StopWordFreeTokens = [token for token in StopWordTokens if token not in StopWordList]

    # Stem all tokens
    StemmedTokens = [WordStemmer.stem(token) for token in StopWordFreeTokens]

    # Lemmatize all tokens
    LemmatizedTokens = [Lemmatizer.lemmatize(token) for token in StemmedTokens]

    # Combine tokens into a string and then split using regex to turn tokens into strings
    TokensString = ' '.join(LemmatizedTokens)
    SplitTokensString = TokensString.split()

    # Contruct two lists to store words and word count for BOW representation
    BOWWords = []
    BOWCount = []

    # Loop through all Strings in SplitTokensString, adding if not present, increment count by 1 if present
    WorkingIndex = 0
    while WorkingIndex < len(SplitTokensString):
        if SplitTokensString[WorkingIndex] in BOWWords:
            CountIndex = BOWWords.index(SplitTokensString[WorkingIndex])
            BOWCount[CountIndex] += 1
        else:
            BOWWords.append(SplitTokensString[WorkingIndex])
            BOWCount.append(1)
        WorkingIndex += 1

    # Return BOW representation as 2 lists to turn into a list of lists for IndividualBOW
    return BOWWords, BOWCount

# Define a function to generate BOW for each document individually and make a list of lists
def IndividualBOW(ReadFile):
    # Create empty lists to store the lists in to make a list of lists
    BOWWordsList = []
    BOWCountList = []
    ClassificationList = []

    # Create an index to loop over every element of the array and use while loop to iterate
    WorkingIndex = 0
    while WorkingIndex < ReadFile.shape[0]:
        # Normalize the corpus currently selected
        WorkingBOWWords, WorkingBOWCount = NormalizeAndBOW(str(ReadFile.iloc[WorkingIndex, 1]))

        # Add BOW generated fromm corpus to lists
        BOWWordsList.append(WorkingBOWWords)
        BOWCountList.append(WorkingBOWCount)

        # Grab the classification of the corpus
        ClassificationList.append(str(ReadFile.iloc[WorkingIndex, 2]))

        # Iterate
        WorkingIndex += 1

    # Return list of lists for DictBind and ClassificationList for train_test_split
    return BOWWordsList, BOWCountList, ClassificationList

# Define a function to bind the BOWWords and BOWCounts together in a dictionary, then make a list of those dictionaries
def DictBind(BOWWordsList, BOWCountList):
    # Declare indices for iteration
    WorkingIndex = 0
    WorkingLineIndex = 0

    # Declare empty list for full dictionary
    FinalDictList = []

    # Iterate over number of BOWs
    while WorkingLineIndex < len(BOWWordsList):

        # Index to iterate over word and counts of each BOW
        WorkingIndex = 0

        # Empty dictionary to be appended to final list
        WorkingDict = {}

        # Iterate over word and counts of each BOW and add to dictionary
        while WorkingIndex < len(BOWWordsList[WorkingLineIndex]):
            # Bind word and count in the BOW
            WorkingDict[BOWWordsList[WorkingLineIndex][WorkingIndex]] = BOWCountList[WorkingLineIndex][WorkingIndex]

            # Iterate to next word and count
            WorkingIndex += 1

        # Append the current dictionary of current BOW to full list of dictionaries
        FinalDictList.append(WorkingDict)

        # Iterate to next BOW
        WorkingLineIndex += 1

    # Return FinalDictList to be turned into a dataframe for train_test_split
    return FinalDictList

# Read PubMed CSV obtained from PubMedAbstractsToCSV.py
InputCSV = pd.read_csv('abstractData.zip')

# Use function to generate BOW
BOWWordsList, BOWCountList, ClassificationList = IndividualBOW(InputCSV)

# Obtain dictionary list using command
DictList = DictBind(BOWWordsList, BOWCountList)

# Turn DictList into a dataframe and fill NaN's with 0's
DFDictList = pd.DataFrame(DictList)
DFDictList.fillna(0, inplace=True)

# Use train_test_split to shuffle data and then take a 7:3 ratio of train to validations
FeaturesTrain, FeaturesTest, TargetsTrain, TargetsTest = train_test_split(DFDictList, ClassificationList, train_size=.7, random_state=42)

# Fit the data
NBModel.fit(FeaturesTrain, TargetsTrain)

# Use NB to predict values
TargetsPred = NBModel.predict(FeaturesTest)

# Print statistics to verify accuracy of model
print("Confusion Matrix:")
print(confusion_matrix(TargetsTest, TargetsPred))
print()
print("Accuracy Score:")
print(accuracy_score(TargetsTest, TargetsPred))
print()
print("Precision Score:")
print(precision_score(TargetsTest, TargetsPred, average='macro'))
print()
print("Recall Score:")
print(recall_score(TargetsTest, TargetsPred, average='macro'))
print()
print("F1 Score:")
print(f1_score(TargetsTest, TargetsPred, average='macro'))
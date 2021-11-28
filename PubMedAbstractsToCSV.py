import pandas as pd
import requests
from bs4 import BeautifulSoup
import Bio
from Bio import Entrez

# Search terms put into format for PubMed database retrieval. [MH] returns MeshTerms, hasabstract filters for only
# articles with abstracts.
# In order: Acute Rheumatic Arthritis, Cardiovascular Abnormalities, Lyme Disease, and Knee Osteoarthritides
searchTerms = ['acute+rheumatic+arthritis[MH] hasabstract',
               'Cardiovascular+Abnormalities[MH] hasabstract',
               'Lyme+Disease[MH] hasabstract',
               'Knee+Osteoarthritides[MH] hasabstract'
               ]

# Email to be contacted if data obtained exceeds PubMed thresholds
Entrez.email = "dominic.polletta@maine.edu"

index = 0

# This method takes the ID list obtained from the ESearch function located in createDatabase,
# joins them together with commas to be in proper syntax for the EFetch method. This method returns
# a list of papers with abstract.
def getDetails(idList):
    ids = ",".join(idList)
    Entrez.email = "dominic.polletta@maine.edu"
    handle = Entrez.efetch(db="pubmed", retmode="xml", id=ids)
    results = Entrez.read(handle)
    return results


def createDatabaseOfPubmedRecord(searchTerms):
    numRecords = []
    abstractTexts = []

    #Performs operation for every search term in list.
    for index in range(len(searchTerms)):
        #Stores at a max of 10000 records, and only returns records published after 2010
        handle = Entrez.esearch(db="pubmed", retmode='xml', retmax=10000, mindate=2010, term=searchTerms[index])
        record = Entrez.read(handle)
        print(record)
        idList = record['IdList']
        listOfPapers = getDetails(idList)

        numPapers = 0;
        for paperIndex, paper in enumerate(listOfPapers['PubmedArticle']):

            #Attempts to retrieve only the abstract from each article. If there is no abstract despite
            #searching for only articles with abstracts, throws an exception and prints "whoopsies!" to the console.
            try:
                abstractTexts.append(paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0])
                numPapers += 1
            except:
                print("whoopsies!")
        numRecords.append(numPapers)

    return abstractTexts, numRecords


abstractTexts, numRecords = createDatabaseOfPubmedRecord(searchTerms)
print(abstractTexts)
print(numRecords)

textClass = ["Acute Rheumatic Arthritis", "Cardiovascular Abnormalities", "Lyme Disease", "Knee Osteoarthritides"]

#Gets total number of records retrieved
totalRecords = 0
for index in range(len(numRecords)):
    totalRecords += numRecords[index]
print('Total Records')
print(totalRecords)
print()

#Gets end of each class of texts' index for classification in the DF
abstractClasses = []
firstClassIndex = numRecords[0]
secondClassIndex = numRecords[0] + numRecords[1]
thirdClassIndex = numRecords[0] + numRecords[1] + numRecords[2]
fourthClassIndex = numRecords[0] + numRecords[1] + numRecords[2] + numRecords[3]
print('Num abstract texts & classes')

#Creates a list of text classes 1 to 1 for each abstract, so it may be compiled into a DF.
for index in range(totalRecords):
    if index <= firstClassIndex:
        abstractClasses.append(textClass[0])
    elif firstClassIndex < index <= secondClassIndex:
        abstractClasses.append(textClass[1])
    elif secondClassIndex < index <= thirdClassIndex:
        abstractClasses.append(textClass[2])
    elif thirdClassIndex < index <= fourthClassIndex:
        abstractClasses.append(textClass[3])

print(len(abstractTexts))
print(len(abstractClasses))

#Creates DF with abstracts in one column and their appropriate class in the next.
abstractDict = {'Abstracts': abstractTexts,
                'Class': abstractClasses
                }
df = pd.DataFrame(abstractDict)

#Saves CSV of DF data
compression_opts = dict(method='zip', archive_name='abstractData.csv')
df.to_csv('abstractData.zip', compression=compression_opts)

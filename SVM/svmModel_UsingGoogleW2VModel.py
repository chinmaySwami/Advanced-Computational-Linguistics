import pandas as pd
import gensim
import re
from nltk.corpus import stopwords

# Read the Training dataset into a pandas dataframe
trainSet = pd.read_csv("../Train.csv");

# Extract sentences from the training set and convert it to a list
sentences = trainSet["sentence"].tolist()
print("Sentence Length", len(sentences))
wordList = []

#  Using regular expression to get rid of non-alphanumeric characters
for sentence in sentences:
    #  converting sentences to list of words
    wordList.append(re.sub("[^\w]", " ", sentence).split())

print("List Length", len(wordList))

# Getting rid of stop words present in the sentences
stop_words = set(stopwords.words('english'))

tempListLowerLevel = []
tempListUpperLevel = []

#  checking if a word is stopword and not a alpha numeric if yes then removing it

# for alist in wordList:
#     tempListLowerLevel = []
#     for word in alist:
#         if word not in stop_words:
#             if word.isalnum() == True:
#                 tempListLowerLevel.append(word)
#     tempListUpperLevel.append(tempListLowerLevel)
#
# wordList = tempListUpperLevel
# print(wordList)
# print("After removing stop words", len(wordList))

# Generate word embeddings using googles pretrained word2vec model
word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(
    '../Google Word2Vec model/GoogleNews-vectors-negative300.bin',binary=True)

# tempListLowerLevel = []
# tempListUpperLevel = []
# count = 0
# for alist in wordList:
#     if count < 5:
#         tempList = []
#         for word in alist:
#             tempListLowerLevel.append(word2VecModel[word])
#         tempListUpperLevel.append(tempListLowerLevel)
#         count+=1

print("MTX:     ",word2VecModel["MTX"])
print("q24h:     ",word2VecModel["q24h"])







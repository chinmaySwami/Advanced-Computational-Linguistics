import pandas as pd
import gensim
import re
from nltk.corpus import stopwords
import fastText
from sklearn import svm
import numpy as np

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

# Converting words to lower case and padding zeros
tempListLower = []
tempListHigher = []
maxSentenceLength = 0

for list in wordList:
    tempListLower = []
    if len(list) > maxSentenceLength:
        maxSentenceLength = len(list)
    for words in list:
        tempListLower.append(words.lower())
    tempListHigher.append(tempListLower)

wordList = tempListHigher
print("List with lower letters length", len(wordList))

# Generating sentence back from list of words

# Load fasttext model
fastTextModel = fastText.load_model('../Fasttrack_model/fil9.bin')

# Generating list of vectors for sentences
tempListLower = []
tempListHigher = []

for alist in wordList:
    tempListLower = []
    for words in alist:
        tempListLower.append(fastTextModel.get_word_vector(words))
    tempListHigher.append(tempListLower)

sentenceVectorList = tempListHigher

# Simple test to check if everything is right
print("Vector sentence representation length ", len(sentenceVectorList))
print("Number of words in 4th sentence: ", len(wordList[4]))
print("Number of Vectors  in 4th sentence: ", len(sentenceVectorList[4]))
print("Length of some lists in the sentenceVectorList", len(sentenceVectorList[6]),
      len(sentenceVectorList[100]), len(sentenceVectorList[2000]))

# Converting Entities to Word2Vec
# For entity 1
entity1 = trainSet["E1"].tolist()
print("Number of entity 1 in dataset ", len(entity1))

tempList = []
for entity in entity1:
    tempList.append(fastTextModel.get_word_vector(entity))

entity1 = tempList
print("Number of entity 1 in dataset after word2vec ", len(entity1))

# For entity 2
entity2 = trainSet["E2"].tolist()
print("Number of entity 2 in dataset ", len(entity2))

tempList = []
for entity in entity2:
    tempList.append(fastTextModel.get_word_vector(entity))

entity2 = tempList
print("Number of entity 1 in dataset after word2vec ", len(entity2))

# For Result Label
typesTrain = trainSet["type"].tolist()
print("Number of types in dataset ", len(typesTrain))
tempList = []
for type in typesTrain:
    tempList.append(fastTextModel.get_word_vector(type))

typesTrain = tempList
print("Number of types in dataset after word2vec ", len(typesTrain))

# Generating training set by merging sentence,entity1 and entity2 vector
trainingSet = []
trainingSet.append(sentenceVectorList)
trainingSet.append(entity1)
trainingSet.append(entity2)
print("Length of training set: ", len(trainingSet))


# Generating pandas dataframe for traning set
print(len(sentenceVectorList), len(entity1), len(entity2))

tempListLower = []
tempListHigher = []
for rowNo in range(0, 26005):
    tempListLower = []
    tempListLower = sentenceVectorList[rowNo] + entity1[rowNo] + entity2[rowNo]
    tempListHigher.append(tempListLower)

trainingSetArray = np.asarray(tempListHigher)
print("Shape: ", trainingSetArray.shape)

# Training SVM model
clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf.fit(tempListHigher, typesTrain)

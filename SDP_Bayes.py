# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
#
#
# #Calling the dataset using pandas library
# dataset1 = pd.read_csv("jEdit_4.0_4.2.csv")
# dataset2 = pd.read_csv("jEdit_4.2_4.3.csv")
#
# #number of class
# print("Dataset 1")
# n_defect_dataset1 = dataset1['Class'][dataset1['Class'] == True].count()
# print("Number of Defect Class:", n_defect_dataset1)
# n_non_defect_dataset1 = dataset1['Class'][dataset1['Class'] == False].count()
# print("Number of Non-Defect Class:", n_non_defect_dataset1)
# total_dataset1 = dataset1['Class'].count()
# print("Total Dataset 1:", total_dataset1)
# P_defect_dataset1 = n_defect_dataset1 / total_dataset1
# print("Probability of Defective Module In Dataset 1:", P_defect_dataset1)
# P_non_defect_dataset1 = n_non_defect_dataset1 / total_dataset1
# print("Probability of Non-Defective Module In Dataset 1:", P_non_defect_dataset1)
# print()
# print("--------------------------------------------------------------")
# print("Dataset 2")
# n_defect_dataset2 = dataset2['Class'][dataset2['Class'] == True].count()
# print("Number of Defect Class:", n_defect_dataset2)
# n_non_defect_dataset2 = dataset2['Class'][dataset2['Class'] == False].count()
# print("Number of Non-Defect Class:", n_non_defect_dataset2)
# total_dataset2 = dataset2['Class'].count()
# print("Total Dataset 2:", total_dataset2)
# P_defect_dataset2 = n_defect_dataset2 / total_dataset2
# print("Probability of Defective Module In Dataset 2:", P_defect_dataset2)
# P_non_defect_dataset2 = n_non_defect_dataset2 / total_dataset2
# print("Probability of Non-Defective Module In Dataset 2:", P_non_defect_dataset2)
# print()
# #Calculating mean based on class
# print("Dataset 1 Mean")
# print(dataset1.mean())
# print(dataset1.groupby("Class").mean())
# print("----------------------------------------------------------------")
# print("Dataset 2 Mean")
# print(dataset2.mean())
# print(dataset2.groupby("Class").mean())
# print("-----------------------------------------------------------------")
# print("Dataset 1 Prediction And Peformance")
# feature_dataset1 = dataset1[['WMC', 'DIT', 'NOC', 'CBO', 'RFC', 'LCOM', 'NPM', 'LOC']]
# target_dataset1 = dataset1['Class']
#
# feature_dataset1_train, feature_dataset1_test, target_dataset1_train, target_dataset1_test = train_test_split(
#     feature_dataset1, target_dataset1, test_size=0.4, random_state=101)
#
# gaussian_bayes_classifier = GaussianNB()
# fitting = gaussian_bayes_classifier.fit(feature_dataset1_train, target_dataset1_train)
# classifier_models = gaussian_bayes_classifier.predict(feature_dataset1_test)
# accuracy_performance = accuracy_score(target_dataset1_test, classifier_models, normalize=True)
# classification_performance = classification_report(target_dataset1_test, classifier_models)
# print(fitting)
# print(classifier_models)
# print()
# print("Accuracy Score:",accuracy_performance)
# print(classification_performance)
# print("------------------------------------------------------------------")
# print("Dataset 2 Prediction And Performance")
# feature_dataset2 = dataset2[['WMC', 'DIT', 'NOC', 'CBO', 'RFC', 'LCOM', 'NPM', 'LOC']]
# target_dataset2 = dataset2['Class']
#
# feature_dataset2_train, feature_dataset2_test, target_dataset2_train, target_dataset2_test = train_test_split(
#     feature_dataset2, target_dataset2, test_size=0.4, random_state=101)
# fitting = gaussian_bayes_classifier.fit(feature_dataset2_train, target_dataset2_train)
# classifier_models = gaussian_bayes_classifier.predict(feature_dataset2_test)
# accuracy_performance = accuracy_score(target_dataset2_test, classifier_models, normalize=True)
# classification_performance = classification_report(target_dataset2_test, classifier_models)
# print(fitting)
# print(classifier_models)
# print()
# print("Accuracy Score:", accuracy_performance)
# print(classification_performance)
import csv
import math
import random

def loadCsv(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

# def splitTest(dataset, splitRatio):
#     trainSize = int(len(dataset) * splitRatio)
#     trainSet = []
#     copy = list(dataset)
#     random.seed(15)
#     while len(trainSet) < trainSize:
#         index = random.randrange(len(copy))
#         trainSet.append(copy.pop(index))
#     return [trainSet, copy]

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return variance

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * stdev)))
    return (1 / (math.sqrt(2 * math.pi* stdev))) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

filename = "jEdit_4.0_4.2.csv"
dataset1 = loadCsv(filename)
#splitRatio = 0.15
print(summarizeByClass(dataset1))
print()
#trainingSet, testSet = splitTest(dataset1, splitRatio)
summaries = summarizeByClass(dataset1)
testdataset = loadCsv("testdata.csv")
predictions = getPredictions(summaries, testdataset)
print("---------------- Dataset 1 ----------------------")
print(predictions)
accuracy = getAccuracy(testdataset, predictions)
print(accuracy)
print()
print("---------------- Dataset 2 ----------------------")
dataset2 = loadCsv("jEdit_4.2_4.3.csv")
#trainingSet2, testSet2 = splitTest(dataset2, splitRatio)
summaries2 = summarizeByClass(dataset2)
predictions2 = getPredictions(summaries2, testdataset)
print(predictions2)
accuracy2 = getAccuracy(testdataset, predictions2)
print(accuracy2)
print()
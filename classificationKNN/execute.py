import kNN
import matplotlib
import matplotlib.pyplot as plt

group, labels = kNN.createDataSet()
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')

normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

kNN.datingClassTest()
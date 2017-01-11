import kNN

group, labels = kNN.createDataSet()
datingDataMat, datingDataLabels = kNN.file2matrix('datingTestSet.txt')

print datingDataMat
print datingDataLabels[0:20]
import pickle


fileName = 'mnist.p'
trainData, trainLabel, one_hot_trainLabel, testData = pickle.load(open(fileName, mode='rb'))


print(type(one_hot_trainLabel[0]))
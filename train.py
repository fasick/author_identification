import csv
import random
from sys import argv
from os import _exit, path, listdir
import tensorflow as tf

class Network:
    def __init__(self, numHiddenLayers, numNodes, learningRate, numTrainingEpochs, optimizeFunction, numCorrectInRowToExit):
        #########################################
        self.numInputs = numNodes[0]
        self.numOutputs = numNodes[-1]
        self.X = tf.placeholder(name='X', dtype=tf.float32, shape=[None, numNodes[0]])
        self.Y = tf.placeholder(name='Y', dtype=tf.float32, shape=[None, numNodes[-1]])
        #########################################
        self.weights = [tf.Variable(tf.random.normal([x,y])) for x,y in zip(numNodes[:-1], numNodes[1:])]
        self.biases  = [tf.Variable(tf.random.normal([x])) for x in numNodes[1:]]
        self.numHiddenLayers = numHiddenLayers
        self.numNodes = numNodes
        #########################################
        self.learningRate = learningRate
        self.numTrainingEpochs = numTrainingEpochs
        self.logits = self.ForwardPropagateModel()
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = optimizeFunction(learning_rate=self.learningRate)
        self.trainingOpts = self.optimizer.minimize(self.loss)
        self.numCorrectInRowToExit = numCorrectInRowToExit
        self.saver = tf.train.Saver()

    def ForwardPropagateModel(self):
        out = 0
        for layer in range(self.numHiddenLayers + 1):
            if out == 0:
                out = tf.add(tf.matmul(self.X, self.weights[layer]), self.biases[layer])
            else:
                out = tf.add(tf.matmul(out, self.weights[layer]), self.biases[layer])
        return out

    def train(self, trainX, trainY, testX, testY):
        zeroCostCount = 0
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for epoch in range(self.numTrainingEpochs):
                _, c = session.run([self.trainingOpts, self.loss],
                                    feed_dict={self.X: trainX, self.Y: trainY})
                if c == 0:
                    zeroCostCount += 1
                    if zeroCostCount >= self.numCorrectInRowToExit:
                        break
                else:
                    zeroCostCount = 0
                print('Epoch: ', '%04d' % (epoch + 1), 'cost={:.4f}'.format(c))
            # test
            prediction = tf.nn.softmax(self.logits)
            correctPrediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.Y, 1))
            # calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
            print('Accuracy: ', accuracy.eval({self.X: testX, self.Y: testY}))
            if input('Would you like to save this network?[Y|n]').lower() == 'y':
                self.saver.save(session, save_path='/'.join(['savedNetwork', input('Enter filename: ')]))

    def predict(self, X, saveFile):
        with tf.Session() as session:
            tf.train.Saver().restore(session, saveFile)
            p = session.run(tf.nn.softmax(self.logits), feed_dict={self.X: X})
            print(p)
            

if __name__ == '__main__':
    #############################################
    # set defaults and create our lists
    numNodesPerHidden = 1500
    numHiddn = 3
    trainX = []
    trainY = []
    testX = []
    testY = []
    savedNetwork = None
    anonymousData = None
    #############################################
    # some possible optimizers to choose from
    optimizers = [tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer,
                  tf.train.ProximalGradientDescentOptimizer]
    
    numNodes = [21]
    for i in range(numHiddn):
        numNodes.append(numNodesPerHidden)
    numNodes.append(3)
    
    #############################################
    # read in our data
    with open('G8/training.csv', mode='r') as _file:
        reader = csv.reader(_file)
        data = list(reader)
        for line in data:
            feature = []
            label = []
            for item in line[:21]:
                feature.append(float(item))
            for item in line[21:]:
                label.append(float(item))
            trainX.append(feature)
            trainY.append(label)
    with open('G8/testing.csv', mode='r') as _file:
        reader = csv.reader(_file)
        data = list(reader)
        for line in data:
            feature = []
            label = []
            for item in line[:21]:
                feature.append(float(item))
            for item in line[21:]:
                label.append(float(item))
            testX.append(feature)
            testY.append(label)
    
    #############################################
    # create our network and train it
    network = Network(numHiddenLayers=numHiddn,
                      numNodes=numNodes,
                      learningRate=0.07,
                      numTrainingEpochs=5000,
                      optimizeFunction=optimizers[0],
                      numCorrectInRowToExit=10)

    if input('Would you like to train a new model? [Y|n] ').lower() == 'y':
        network.train(trainX, trainY, testX, testY)
    else:
        textFile = None
        while savedNetwork is None:
            savedNetwork = input('Enter directory of saved network: ')
            # try:
            #     listdir(savedNetwork)
            # except NotADirectoryError:
            #     print(savedNetwork, ' is not a directory.')
            #     savedNetwork = None
            # except FileNotFoundError:
            #     print('the directory ', savedNetwork, ' does not exist')
            #     savedNetwork = None
        while textFile is None:
            textFile = input('Enter text file to identify the author: ')
            try:
                with open(textFile, mode='r') as _file:
                    reader = csv.reader(_file)
                    anonymousData = list(reader)
            except FileNotFoundError:
                print('The file ', textFile, ' does not exist')
                textFile = None
        network.predict(anonymousData, savedNetwork)
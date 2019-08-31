from sys import argv
from os import listdir
import tensorflow as tf

# saver = tf.train.Saver()

def predict(fileName):
    with tf.Session() as session:
        tf.train.Saver().restore(session, fileName)
        print('session restored')

if __name__ == '__main__':
    savedNetwork = None
    textFile = None
    anonymousSample = None

    while savedNetwork is None:
        savedNetwork = input('Enter directory of saved network: ')
        try:
            listdir(savedNetwork)
        except NotADirectoryError:
            print(savedNetwork, ' is not a directory.')
            savedNetwork = None
        except FileNotFoundError:
            print('the directory ', savedNetwork, ' does not exist')
            savedNetwork = None
    while textFile is None:
        textFile = input('Enter text file to identify the author: ')
        try:
            with open(textFile, mode='r') as _file:
                anonymousSample = _file.read()
        except FileNotFoundError:
            print('The file ', textFile, ' does not exist')
            textFile = None

    predict(savedNetwork)
import inference
import numpy as np
import sys
import train
import data


def trainNetworks(inputFile, epochs):
    
    # Load data
    # train_data = np.array([[1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.double)
    # train_labels = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.double)
    
    train_data, train_labels, val_data, val_labels, test_data, test_labels = data.read_data(inputFile)

    train.trainNetworks(train_data, train_labels, epochs=epochs)


def runInference(inputFile):
    # TODO: Read from a file
    inputData = np.array([[1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])

    inference.runInference(inputData)


if __name__ == '__main__':
    
    # To run training or inference
    toTrain = sys.argv[1] == 'train'

    if toTrain:
        inputFile = sys.argv[2]
        epochs = int(sys.argv[3])
        trainNetworks(inputFile, epochs)
    else:
        inputFile = sys.argv[2]
        runInference(inputFile)
    

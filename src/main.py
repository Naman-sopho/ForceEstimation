import numpy as np
import sys
import train

def trainNetworks(epochs):
    
    # Load data
    train_data = np.array([[1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.double)
    train_labels = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.double)
    
    train.trainNetworks(train_data, train_labels, epochs=epochs)

if __name__ == '__main__':
    
    # To run training or inference
    toTrain = sys.argv[1] == 'train'
    epochs = int(sys.argv[2])
    
    if toTrain:
        trainNetworks(epochs)
    
    

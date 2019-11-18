import sys
import train

def trainNetworks():
    
    # Load data
    train_data = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train_labels = [1, 2, 3, 4, 5, 6]
    
    train.trainNetworks(train_data, train_labels, epochs=3)

if __name__ == '__main__':
    
    # To run training or inference
    toTrain = sys.argv[1] == 'train'
    
    if toTrain:
        trainNetworks()
    
    
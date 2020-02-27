import matplotlib.pyplot as plt 
import numpy as np 

def plot():
    test_predictions = np.loadtxt('../inference_results/test_predictions.csv', delimiter=',')
    test_labels = np.loadtxt('../inference_results/test_labels.csv', delimiter=',')

    avg_train_loss = np.loadtxt('../train_results/avg_train_loss.csv', delimiter=',')
    avg_val_loss = np.loadtxt('../train_results/avg_val_loss.csv', delimiter=',')

    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    ax1.plot([x for x in range(len(test_predictions))], test_predictions, label='Test Predictions')
    ax1.plot([x for x in range(len(test_labels))], test_labels, label='Test Labels')

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Example')

    ax2.plot([x for x in range(len(test_predictions))], np.subtract(test_predictions, test_labels), label='Difference')
    
    fig1.savefig('error.png')

    fig2, ax3 = plt.subplots()

    ax3.plot([x for x in range(len(avg_train_loss))], avg_train_loss, label='Avg Train Loss')
    ax3.plot([x for x in range(len(avg_val_loss))], avg_val_loss, label='Avg Val Loss')

    ax3.set_xlabel('Loss')
    ax3.set_ylabel('Epoch')

    fig2.savefig('loss.png')
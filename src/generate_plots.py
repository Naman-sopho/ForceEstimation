import matplotlib.pyplot as plt 
import numpy as np 

def plot():
    test_predictions = np.loadtxt('../inference_results/test_predictions.csv', delimiter=',')
    test_labels = np.loadtxt('../inference_results/test_labels.csv', delimiter=',')

    avg_train_loss = np.loadtxt('../train_results/avg_train_loss.csv', delimiter=',')
    avg_val_loss = np.loadtxt('../train_results/avg_val_loss.csv', delimiter=',')

    fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1)

    x = [x for x in range(len(test_predictions))]

    ax1.plot(x, test_predictions[:,0], label='Test Predictions')
    ax1.plot(x, test_labels[:,0], label='Test Labels')

    ax2.plot(x, test_predictions[:,1], label='Test Predictions')
    ax2.plot(x, test_labels[:, 0], label='Test Labels')

    ax3.plot(x, test_predictions[:, 2], label='Test Predictions')
    ax3.plot(x, test_labels[:, 2], label='Test Labels')

    ax4.plot(x, test_predictions[:, 3], label='Test Predictions')
    ax4.plot(x, test_labels[:, 3], label='Test Labels')

    ax5.plot(x, test_predictions[:, 4], label='Test Predictions')
    ax5.plot(x, test_labels[:, 4], label='Test Labels')

    ax6.plot(x, test_predictions[:, 5], label='Test Predictions')
    ax6.plot(x, test_labels[:, 5], label='Test Labels')

    ax1.set_ylabel('Value Joint 1')
    ax1.set_xlabel('Example Number')
    ax2.set_ylabel('Value Joint 2')
    ax3.set_ylabel('Value Joint 3')
    ax4.set_ylabel('Value Joint 4')
    ax5.set_ylabel('Value Joint 5')
    ax6.set_ylabel('Value Joint 6')

    # ax2.plot([x for x in range(len(test_predictions))], np.subtract(test_predictions, test_labels), label='Difference')
    
    fig1.savefig('error.png')

    fig2, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, ncols=1)

    x = [x for x in range(len(avg_train_loss))]

    ax1.plot(x, avg_train_loss[:, 0], label='Avg Train Loss')
    ax1.plot(x, avg_val_loss[:, 0], label='Avg Val Loss')

    ax2.plot(x, avg_train_loss[:, 1], label='Avg Train Loss')
    ax2.plot(x, avg_val_loss[:, 1], label='Avg Val Loss')

    ax3.plot(x, avg_train_loss[:, 2], label='Avg Train Loss')
    ax3.plot(x, avg_val_loss[:, 2], label='Avg Val Loss')

    ax4.plot(x, avg_train_loss[:, 3], label='Avg Train Loss')
    ax4.plot(x, avg_val_loss[:, 3], label='Avg Val Loss')

    ax5.plot(x, avg_train_loss[:, 4], label='Avg Train Loss')
    ax5.plot(x, avg_val_loss[:, 4], label='Avg Val Loss')

    ax6.plot(x, avg_train_loss[:, 5], label='Avg Train Loss')
    ax6.plot(x, avg_val_loss[:, 5], label='Avg Val Loss')

    ax1.set_ylabel('Loss Joint 1')
    ax1.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Joint 2')
    ax3.set_ylabel('Loss Joint 3')
    ax4.set_ylabel('Loss Joint 4')
    ax5.set_ylabel('Loss Joint 5')
    ax6.set_ylabel('Loss Joint 6')

    fig2.savefig('loss.png')
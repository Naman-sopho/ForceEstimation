import network
import torch
import os
import numpy as np

np.set_printoptions(precision=6, suppress=True)

def runInference(input_data, input_labels, input_file=None):
    ACC_DIFF = 0.00001

    input_data = torch.from_numpy(input_data)
    input_data = input_data.type(torch.FloatTensor)

    input_labels = torch.from_numpy(input_labels)
    input_labels = input_labels.type(torch.FloatTensor)

    # Load networks
    MODELS_DIR = os.path.join('../', 'models/')

    network1 = network.model()
    network1.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'network1.torch')))

    network2 = network.model()
    network2.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'network2.torch')))

    network3 = network.model()
    network3.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'network3.torch')))

    network4 = network.model()
    network4.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'network4.torch')))

    network5 = network.model()
    network5.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'network5.torch')))

    network6 = network.model()
    network6.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'network6.torch')))

    correct = np.zeros((6, 1))

    predicted = np.empty((0, 6))
    labels = np.empty((0, 6))

    for data, label in zip(input_data, input_labels):
        output1 = network1(data)
        output2 = network2(data)
        output3 = network3(data)
        output4 = network4(data)
        output5 = network5(data)
        output6 = network6(data)

        predicted = np.append(predicted, np.array([[output1.item(), output2.item(), output3.item(), output4.item(), output5.item(), output6.item()]]), axis=0)
        labels = np.append(labels, np.array([[label[0], label[1], label[2], label[3], label[4], label[5]]]), axis=0)

        correct[0] = correct[0] + 1 if output1.item() - label[0] < ACC_DIFF else correct[0]
        correct[1] = correct[1] + 1 if output2.item() - label[1] < ACC_DIFF else correct[1]
        correct[2] = correct[2] + 1 if output3.item() - label[2] < ACC_DIFF else correct[2]
        correct[3] = correct[3] + 1 if output4.item() - label[3] < ACC_DIFF else correct[3]
        correct[4] = correct[4] + 1 if output5.item() - label[4] < ACC_DIFF else correct[4]
        correct[5] = correct[5] + 1 if output6.item() - label[5] < ACC_DIFF else correct[5]


    np.savetxt('../inference_results/test_predictions.csv', predicted, fmt='%10.10f', delimiter=',')
    np.savetxt('../inference_results/test_labels.csv', labels, fmt='%10.10f', delimiter=',')
    print('Inference Results: {}'.format(np.divide(correct, len(input_data)) *  100))

import network
import torch
import os


def trainNetworks(train_data, train_labels, val_data, val_labels, steps, learning_rate):
    """
    Trains 6 networks on the data using the hyperparameters provided.
    Saves the final model as in the models directory with timestamp as the name.

    Input is same for all 6 networks. The prediction of each network is for its corresponding
    joint. The labels data structure is divided into

    :param train_data: Numpy array with training data points
    :param val_data: Numpy array with validation data points
    :param steps: Number of training steps
    :param learning_rate: Learning rate used for training
    """

    network1 = network.model()
    network2 = network.model()
    network3 = network.model()
    network4 = network.model()
    network5 = network.model()
    network6 = network.model()

    optimiser1 = torch.optim.Adam(network1.parameters(), learning_rate)
    optimiser2 = torch.optim.Adam(network2.parameters(), learning_rate)
    optimiser3 = torch.optim.Adam(network3.parameters(), learning_rate)
    optimiser4 = torch.optim.Adam(network4.parameters(), learning_rate)
    optimiser5 = torch.optim.Adam(network5.parameters(), learning_rate)
    optimiser6 = torch.optim.Adam(network6.parameters(), learning_rate)

    criterion = torch.nn.MSELoss()

    # SGD Implementation
    for epoch in range(epochs):
        for data, label in zip(train_data, train_labels):
            output1 = network1(data)
            output2 = network2(data)
            output3 = network3(data)
            output4 = network4(data)
            output5 = network5(data)
            output6 = network6(data)

            # Each label frame consists of 6 values corresponding to each joint
            loss1 = criterion(output1, label[0])
            loss2 = criterion(output2, label[1])
            loss3 = criterion(output3, label[2])
            loss4 = criterion(output4, label[3])
            loss5 = criterion(output5, label[4])
            loss6 = criterion(output6, label[5])

            # Backpropagate loss through each network and take gradient step
            optimiser1.zero_grad()
            loss1.backward()
            optimiser1.step()

            optimiser2.zero_grad()
            loss2.backward()
            optimiser2.step()

            optimiser3.zero_grad()
            loss3.backward()
            optimiser3.step()

            optimiser4.zero_grad()
            loss4.backward()
            optimiser4.step()

            optimiser5.zero_grad()
            loss5.backward()
            optimiser5.step()

            optimiser6.zero_grad()
            loss6.backward()
            optimiser6.step()

            # Log progress
            print('Step: {}, Loss: [{.3f}, {.3f}, {.3f}, {.3f}, {.3f}, {.3f}]'
                  .format(epoch+1, loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss5.item()))
                  
    # Save the models after training
    MODELS_DIR = os.path.join('..' + 'models')
    
    torch.save(network1.state_dict(), os.path.join(MODELS_DIR, 'network1.torch'))
    torch.save(network1.state_dict(), os.path.join(MODELS_DIR, 'network2.torch'))
    torch.save(network1.state_dict(), os.path.join(MODELS_DIR, 'network3.torch'))
    torch.save(network1.state_dict(), os.path.join(MODELS_DIR, 'network4.torch'))
    torch.save(network1.state_dict(), os.path.join(MODELS_DIR, 'network5.torch'))
    torch.save(network1.state_dict(), os.path.join(MODELS_DIR, 'network6.torch'))
    
            

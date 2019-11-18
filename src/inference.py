import network
import torch
import os

def runInference(inputData):
    inputData = torch.from_numpy(inputData)
    inputData = inputData.type(torch.FloatTensor)

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

    output1 = network1(inputData)
    output2 = network2(inputData)
    output3 = network3(inputData)
    output4 = network4(inputData)
    output5 = network5(inputData)
    output6 = network6(inputData)

    print('Inference Results: {} {} {} {} {} {}'.format(output1.item(), output2.item(), output3.item(), output4.item(), output5.item(), output6.item()))

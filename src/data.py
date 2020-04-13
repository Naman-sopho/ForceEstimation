########################
# Author: Naman Tiwari
# Created: 2/3/20
########################

import numpy as np
import os

def read_data(filename):
    """
    Reads the relevant topics from a given rosbag and returns
    them as numpy arrays.

    :param filename: Name of the rosbag file
    """

    # If csv file present read from it
    data_csv = 'data.csv'
    labels_csv = 'labels.csv'

    if os.path.exists(data_csv):
        data = np.loadtxt(data_csv, delimiter=',')
        labels = np.loadtxt(labels_csv, delimiter=',')

        train_data, train_labels, test_data, test_labels = split_train_test(data, labels)

        return train_data, train_labels, test_data, test_labels

    from rosbag import bag
    bag_ = bag.Bag(filename)

    # The topic which contains data required for training the networks
    joint_topic = '/dvrk/PSM1/state_joint_current'
    jacobian_topic = '/dvrk/PSM1/jacobian_spatial'
    force_topic = '/dvrk/PSM1/atinetft/wrench'

    data = np.empty((0, 12))
    labels = np.empty((0, 6))
    jacobains = np.empty((0, 6, 6))
    forces = np.empty((0,3))

    print("Start reading bag file.")

    for message in bag_.read_messages(topics=joint_topic):
        positions = np.array(message.message.position)
        velocities = np.array(message.message.velocity)

        positions = np.append(positions, velocities)

        efforts = np.array(message.message.effort)

        data = np.append(data, np.array([positions]), axis=0)
        labels = np.append(labels, np.array([efforts]), axis=0)
    
    for message in bag_.read_messages(topics=jacobian_topic):
        jacobian = np.array(message.message.data).reshape((6,6))
        jacobians = np.append(jacobians, np.array([jacobian]), axis=0)

    for message in bag_.read_messages(topics=force_topic):
        force = message.message.wrench.force
        forces = np.append(forces, np.array([[force.x, force.y, force.z]]), axis=0)


    print()
    print("Reading file complete.")

    print("Saving to csv file.")

    np.savetxt('data.csv', data, fmt='%10.10f', delimiter=',')
    np.savetxt('labels.csv', labels, fmt='%10.10f', delimiter=',')
    np.savetxt('jacobians.csv', labels, fmt='%10.10f', delimiter=',')
    np.savetxt('forces.csv', labels, fmt='%10.10f', delimiter=',')

    train_data, train_labels, test_data, test_labels = split_train_test(data, labels)

    return train_data, train_labels, test_data, test_labels

def split_train_test(data, labels):

    # randomly shuffle data and labels
    length = len(data)

    random_perm = np.random.permutation(length)
    data = data[random_perm]
    labels = labels[random_perm]

    # train-test split 80:20
    train_data = data[:int(0.8*length),:]
    train_labels = labels[:int(0.8*length),:]
    test_data = data[int(0.8*length):,:]
    test_labels = labels[int(0.8*length):,:]

    # train-validation split 80:20
    # train_len = length
    # val_data = train_data[:int(0.2*train_len),:]
    # val_labels = train_labels[:int(0.2*train_len,:)]

    # train_data = train_data[int(0.2*train_len):,:]
    # train_labels = train_labels[int(0.2*train_len):,:]

    return train_data, train_labels, test_data, test_labels

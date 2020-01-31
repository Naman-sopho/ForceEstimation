from rosbag import bag
import numpy as np

def read_data(filename):
    """
    Reads the relevant topics from a given rosbag and returns
    them as numpy arrays.

    :param filename: Name of the rosbag file
    """

    bag_ = bag.Bag(filename)

    # The topic which contains data required for training the networks
    relevant_topic = 'state_joint_current'

    data = np.empty((0, 12))
    labels = np.empty((0, 6))

    print("Start reading bag file.")

    for message in bag_.read_messages():
        if relevant_topic in message.topic:
            positions = np.array(message.message.position)
            velocities = np.array(message.message.velocity)

            positions = np.append(positions, velocities)

            efforts = np.array(message.message.effort)

            data = np.append(data, np.array([positions]), axis=0)
            labels = np.append(labels, np.array([efforts]), axis=0)

    print()
    print("Reading file complete.")

    print("Saving to csv file.")

    np.savetxt('data.csv', data, fmt='%10.10f', delimiter=',')
    np.savetxt('labels.csv', labels, fmt='%10.10f', delimiter=',')

    return data, labels
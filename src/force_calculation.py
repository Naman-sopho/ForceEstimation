import numpy as np

def calcForceFromPredictions(predictions, jacobians, forces):
    print("Starting force calculations...")

    results = np.empty((0,6))
    for (prediction, jacobian) in zip(predictions, jacobians):
        jacobian = jacobian.reshape((6,6))
        jacobian_inv = np.linalg.inv(jacobian)

        result = jacobian_inv.dot(prediction.transpose())
        result = result.transpose()

        results = np.append(results, np.array([result]), axis=0)

    print("Saving force results...")
    np.savetxt('../inference_results/test_predicted_forces.csv', results, fmt='%10.10f', delimiter=',')
    np.savetxt('../inference_results/test_actual_forces.csv', forces, fmt='%10.10f',delimiter=',')
    


if __name__ == '__main__':
    jacobians = np.loadtxt('jacobians.csv', delimiter=',')
    forces = np.loadtxt('forces.csv', delimiter=',')
    predictions = np.array([1,2,3,4,5,6])
    calc_force_from_predictions(predictions, jacobians, forces)

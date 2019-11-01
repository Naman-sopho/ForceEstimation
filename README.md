# ForceEstimation
Neural Network based external force estimation on dVRK.

Network implementation for real time application, based on the architecture defined in the paper titled "Neural Network based Inverse Dynamics Identification and External Force Estimation on the da Vinci Research Kit" by Nural Yilmaz et al.

## Network architecture
#### Layer 1
Input Size: 12, one set of Position and Velocity measurement from each of the 6 joints.  
Output Size: 100  
Activation: `tanh`  
  
#### Layer 2
Input Size: 100  
Output Size: 1  
  
  
6 such networks are used. Each network is trained to provide a torque estimate of one the joints. This torque is then used for the Force estimate.

#!/usr/bin/env python

import rospy
import numpy as np
from sklearn.externals import joblib
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def basis(x1, x2, x3, x4, x5, x6, x7, x8, L1, L2, L3, L4, L5, L6, L7, L8):
  return np.array([1, x1, x2, x3, x4, x5, x6, x7, x8])

def basis_dx():
  return np.matrix([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
    ])

def basis_du():
  return np.matrix([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1]
    ])

def load_data(filepath):

  history = joblib.load(filepath)
  history = history.T

  xs = history[:,0]
  ys = history[:,1]
  thetas = history[:,2]
  vel_xs = history[:,3]
  vel_ys = history[:,4]
  vel_thetas = history[:,5]
  samples = history[:,14]
  goal_xs = history[:,15]
  goal_ys = history[:,16]

  goal_thetas = [1*np.pi/2.0]*len(goal_ys)

  xs_goal_centered = xs - goal_xs
  ys_goal_centered = ys - goal_ys
  thetas_goal_centered = thetas - goal_thetas

  thruster_main = history[:,6]
  thruster_side = history[:,7]

  num_samples = samples[-1]

  Z = np.array([xs_goal_centered, ys_goal_centered, thetas_goal_centered, vel_xs, vel_ys, vel_thetas, thruster_main, thruster_side])
  X,Y = fix_data(Z, samples)

  return X, Y, num_samples

def fix_data(Z, samples):
  X = np.zeros((Z.shape[0],0))
  Y = np.zeros((Z.shape[0],0))
  for trial_idx in range(1, int(samples[-1])):
    idxs = np.where(samples == trial_idx)[0]
    trial_values = Z[:,idxs[0]:idxs[-1]+1]
    X = np.append(X, trial_values[:,0:-1], 1)
    Y = np.append(Y, trial_values[:,1:], 1)
  return X,Y

def main():

  # initalize node
  rospy.init_node('compute_koopman')

  NUM_STATES = 8

  # initialize parameters
  data_path = rospy.get_param('/compute_koopman/data_path')

  L1, L2, L3, L4 = 1.0, 1.0, 1.0, 1.0
  L5, L6, L7, L8 = 1.0, 1.0, 1.0, 1.0

  LS = np.array([L1, L2, L3, L4, L5, L6, L7, L8])
  LS = np.reshape(LS, (LS.shape[0],1))

  # load data
  filename_in = data_path + 'data.pkl'
  X, Y, num_samples = load_data(filename_in)

  # compute koopman
  nkt = basis(0,0,0,0,0,0,0,0,L1,L2,L3,L4,L5,L6,L7,L8)
  nk = nkt.shape[0] # size of basis function
  G = np.zeros((nk,nk))
  A = np.zeros((nk,nk))
  for j in range(X.shape[1]):
    Phi_xm = np.mat(basis(X[0,j],X[1,j],X[2,j],X[3,j],X[4,j],X[5,j],X[6,j],X[7,j],L1,L2,L3,L4,L5,L6,L7,L8))
    Phi_ym = np.mat(basis(Y[0,j],Y[1,j],Y[2,j],Y[3,j],Y[4,j],Y[5,j],Y[6,j],Y[7,j],L1,L2,L3,L4,L5,L6,L7,L8))
    G = G + (1/float(num_samples))*Phi_xm.T*Phi_xm
    A = A + (1/float(num_samples))*Phi_xm.T*Phi_ym

  K = np.linalg.pinv(G)*A

  dphi_dx = basis_dx()
  dphi_du = basis_du()

  A = K[:,1:NUM_STATES+1].T*dphi_dx
  B = K[:,1:NUM_STATES+1].T*dphi_du

  filename_a = data_path + 'A.pkl'
  filename_b = data_path + 'B.pkl'
  filename_k = data_path + 'K.pkl'

  joblib.dump(A, filename_a)
  joblib.dump(B, filename_b)
  joblib.dump(K, filename_k)
  print('saved Koopman model and A and B matrices')

if __name__=='__main__':
  main()

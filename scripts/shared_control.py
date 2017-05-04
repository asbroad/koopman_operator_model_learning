#!/usr/bin/env python

import gym
import rospy
from sensor_msgs.msg import Joy
from sklearn.externals import joblib
from pyglet.window import key
import numpy as np
import scipy
import random
import time
np.set_printoptions(precision=4, suppress=True, linewidth=120)

class LunarLander():

  def __init__(self):
    # initalize node
    rospy.init_node('shared_control')

    NUM_STATES = 6

    # register shutdown hook
    rospy.on_shutdown(self.shutdown_hook)
    self.called_shutdown = False

    # get parameters
    self.data_path = rospy.get_param('/lunar_lander/data_path')
    self.main_joystick = rospy.get_param('/lunar_lander/main_joystick')
    self.inverted = rospy.get_param('/lunar_lander/inverted')

    # set up LQR
    filename_a = self.data_path + 'A.pkl'
    filename_b = self.data_path + 'B.pkl'
    self.A = joblib.load(filename_a)
    self.B = joblib.load(filename_b)

    self.A = self.A[0:NUM_STATES,:]
    self.B = self.B[0:NUM_STATES,:]

    # define weighting structure
    Q = np.diag([1, 1, 1, 1, 1, 1])
    R = np.diag([1, 1])

    P = scipy.linalg.solve_discrete_are(self.A, self.B, Q, R)
    self.K = np.matrix(scipy.linalg.inv(self.B.T*P*self.B+R)*(self.B.T*P*self.A))

    # keep track of current keystroke
    self.user_actions = [0, 0]
    self.terminate = False

    # build environment
    self.env = gym.make('LunarLanderMultiFire-v0')
    self.env.reset()
    self.env.render()

    # set up keystroke hooks
    self.env.viewer.window.on_key_press = self.key_press

    # set up joy subscriber
    rospy.Subscriber('/joy', Joy, self.joy_callback)

    # set up goal locations
    self.goal_x_list = [10]
    self.goal_y_list = [6]
    self.goal_x_idx = 0

    # run system with input from user
    r = rospy.Rate(10)
    self.total_reward, self.total_steps, self.trial_steps, self.trial_idx = 0, 0, 0, 1
    self.success = 0
    self.start_time = time.time()
    while not rospy.is_shutdown():
      if self.check_if_success():
        self.success = 1
        self.success = 0
        self.env.reset()
      if self.env.legs[0].ground_contact or self.env.legs[1].ground_contact:
        self.env.reset()
      if self.terminate == True:
        self.shutdown_hook()
        print('Terminating early')
        break

      # get current state
      cur_state = np.array([
        self.env.lander.position.x - self.goal_x_list[0],
        self.env.lander.position.y - self.goal_y_list[0],
        self.env.lander.angle,
        self.env.lander.linearVelocity.x,
        self.env.lander.linearVelocity.y,
        self.env.lander.angularVelocity
        ])

      # get LQR control
      cur_state = cur_state.reshape((6,1))
      opt_u = -self.K*cur_state
      opt_u_list = [opt_u.item(0), opt_u.item(1)]

      # saturate LQR control
      opt_u_list_saturated = [max(min(x, 1), -1) for x in opt_u_list]

      # get user control
      main_thruster, side_thruster = self.user_actions

      # combine optimal control and user control
      # if sign agrees, let signal through, if not, then don't
      main_thruster_combined = 0
      side_thruster_combined = 0
      if np.sign(main_thruster) == np.sign(opt_u_list_saturated[0]):
        main_thruster_combined = main_thruster
      if np.sign(side_thruster) == np.sign(opt_u_list_saturated[1]):
        side_thruster_combined = side_thruster

      # combine to create input to system
      joined_input = np.array([main_thruster_combined, side_thruster_combined])

      observation, reward, done, info = self.env.step(joined_input)
      self.total_reward += reward
      self.total_steps += 1
      self.trial_steps += 1

      if done:
        self.env.reset()

      # update screen and keep time
      self.env.render()
      r.sleep()

  def check_if_success(self):
    dist = np.sqrt(np.power((self.env.lander.position.x - self.env.goal_x), 2) + np.power((self.env.lander.position.y - self.env.goal_y), 2))
    x_vel = np.sqrt(np.power(self.env.lander.linearVelocity.x, 2))
    y_vel = np.sqrt(np.power(self.env.lander.linearVelocity.y,2))
    a_vel = np.sqrt(np.power(self.env.lander.angularVelocity,2))
    if dist < 0.9 and x_vel < 1 and y_vel < 1 and a_vel < 0.3:
      return True
    else:
      return False

  def shutdown_hook(self):
    if not self.called_shutdown:
      print('Shutting down')

  def joy_callback(self, data):
    invert = 1
    if self.inverted:
      invert = -1
    if self.main_joystick == 'right':
      self.user_actions = [data.axes[3], invert*data.axes[0]]
    elif self.main_joystick == 'left':
      self.user_actions = [data.axes[1], invert*data.axes[2]]

  def key_press(self, k, mod):
    if k == key.SPACE:
      self.terminate = True

if __name__=='__main__':
  ll = LunarLander()

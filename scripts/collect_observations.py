#!/usr/bin/env python

import gym
import rospy
from sensor_msgs.msg import Joy
from sklearn.externals import joblib
from pyglet.window import key
import numpy as np
import random

class LunarLander():

  def __init__(self):
    # initalize node
    rospy.init_node('collect_observations')

    # register shutdown hook
    rospy.on_shutdown(self.shutdown_hook)
    self.called_shutdown = False

    self.data_path = rospy.get_param('/lunar_lander/data_path')
    self.main_joystick = rospy.get_param('/lunar_lander/main_joystick')
    self.inverted = rospy.get_param('/lunar_lander/inverted')

    # keep track of current keystroke
    self.user_actions = [0, 0]

    # gather data for storage
    self.history_pos_x = []
    self.history_pos_y = []
    self.history_pos_theta = []
    self.history_vel_x = []
    self.history_vel_y = []
    self.history_vel_theta = []
    self.history_user_action = []
    self.history_main_thruster = []
    self.history_side_thruster = []
    self.history_init_x = []
    self.history_init_y = []
    self.history_init_ang = []
    self.history_landing_pad_left = []
    self.history_landing_pad_right = []
    self.history_landing_pad_y = []
    self.history_goal_x = []
    self.history_goal_y = []
    self.trial_idxs = []

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
    self.total_reward, self.total_steps, self.trial_idx = 0, 0, 1
    while not rospy.is_shutdown():
      if self.check_if_success():
        self.restart_trial_fixes()
        self.trial_idx += 1
        self.env.reset()
      if self.env.legs[0].ground_contact or self.env.legs[1].ground_contact:
        self.restart_trial_fixes()
        self.trial_idx += 1
        self.env.reset()
      if self.terminate == True:
        self.shutdown_hook()
        print('Terminating early')
        break
      # get user input
      main_thruster, side_thruster = self.user_actions
      observation, reward, done, info = self.env.step(np.array([main_thruster, side_thruster]))
      self.total_reward += reward
      self.total_steps += 1
      # store history data
      self.history_pos_x.append(self.env.lander.position.x)
      self.history_pos_y.append(self.env.lander.position.y)
      self.history_pos_theta.append(self.env.lander.angle)
      self.history_vel_x.append(self.env.lander.linearVelocity.x)
      self.history_vel_y.append(self.env.lander.linearVelocity.y)
      self.history_vel_theta.append(self.env.lander.angularVelocity)
      self.history_main_thruster.append(main_thruster)
      self.history_side_thruster.append(side_thruster)
      # the values below are always the same
      self.history_init_x.append(self.env.INITIAL_POS_X)
      self.history_init_y.append(self.env.INITIAL_POS_Y)
      self.history_init_ang.append(self.env.INITIAL_POS_ANG)
      self.history_landing_pad_left.append(self.env.helipad_x1)
      self.history_landing_pad_right.append(self.env.helipad_x2)
      self.history_landing_pad_y.append(self.env.helipad_y)
      # this value only updates after restarting the trial
      self.trial_idxs.append(self.trial_idx)
      self.history_goal_x.append(self.env.goal_x)
      self.history_goal_y.append(self.env.goal_y)

      if done:
        self.restart_trial_fixes()
        self.trial_idx += 1
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

  def restart_trial_fixes(self):
    num_to_remove = 2
    # update stored values
    self.total_steps -= num_to_remove
    self.history_pos_x = self.history_pos_x[:-num_to_remove]
    self.history_pos_y = self.history_pos_y[:-num_to_remove]
    self.history_pos_theta = self.history_pos_theta[:-num_to_remove]
    self.history_vel_x = self.history_vel_x[:-num_to_remove]
    self.history_vel_y = self.history_vel_y[:-num_to_remove]
    self.history_vel_theta = self.history_vel_theta[:-num_to_remove]
    self.history_main_thruster = self.history_main_thruster[:-num_to_remove]
    self.history_side_thruster = self.history_side_thruster[:-num_to_remove]
    self.history_init_x = self.history_init_x[:-num_to_remove]
    self.history_init_y = self.history_init_y[:-num_to_remove]
    self.history_init_ang = self.history_init_ang[:-num_to_remove]
    self.history_landing_pad_left = self.history_landing_pad_left[:-num_to_remove]
    self.history_landing_pad_right = self.history_landing_pad_right[:-num_to_remove]
    self.history_landing_pad_y = self.history_landing_pad_y[:-num_to_remove]
    self.trial_idxs = self.trial_idxs[:-num_to_remove]
    self.history_goal_x = self.history_goal_x[:-num_to_remove]
    self.history_goal_y = self.history_goal_y[:-num_to_remove]
    # update goal
    self.goal_x_idx += 1
    self.env.goal_x = self.goal_x_list[(self.goal_x_idx)%len(self.goal_x_list)]
    self.env.goal_y = random.choice(self.goal_y_list)

  def shutdown_hook(self):
    if not self.called_shutdown:
      print('Shutting down')
      self.called_shutdown = True
      self.history = np.zeros((17, self.total_steps))
      self.history[0,:] = self.history_pos_x
      self.history[1,:] = self.history_pos_y
      self.history[2,:] = self.history_pos_theta
      self.history[3,:] = self.history_vel_x
      self.history[4,:] = self.history_vel_y
      self.history[5,:] = self.history_vel_theta
      self.history[6,:] = self.history_main_thruster
      self.history[7,:] = self.history_side_thruster
      self.history[8,:] = self.history_init_x
      self.history[9,:] = self.history_init_y
      self.history[10,:] = self.history_init_ang
      self.history[11,:] = self.history_landing_pad_left
      self.history[12,:] = self.history_landing_pad_right
      self.history[13,:] = self.history_landing_pad_y
      self.history[14,:] = self.trial_idxs
      self.history[15,:] = self.history_goal_x
      self.history[16,:] = self.history_goal_y

      filename = self.data_path + 'data.pkl'

      joblib.dump(self.history, filename)
      print('Saved data')

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

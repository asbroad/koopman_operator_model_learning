# Learning Models for Shared Control of Human-Machine Systems with Unknown Dynamics

For detailed information about this system and the theory behind it, please read our [corresponding paper](https://rssrobotics.github.io/program/papers/50/).  The code in this repository can be used to replicate the experiments described in our work.

### Installation

1. Create a ROS workspace
```Shell
  mkdir -p ~/ros_ws/src
  cd ~/ros_ws/src
  catkin_init_workspace
  ```
2. Clone the Koopman Operator Model Learning repository in your ROS workspace
```Shell
  git clone https://github.com/asbroad/koopman_operator_model_learning.git
  ```
3. Clone our modified Gym repository in your ROS workspace
```Shell
  git clone https://github.com/asbroad/gym.git
  ```
4. From the base directory of the workspace, build the code
```Shell
  cd ~/ros_ws
  catkin_make
  ```

### Description of the System

Our shared control framework relies on a learned model of the system and control dynamics.  Therefore, to use the shared control system, you first need to collect demonstration data which can be used to learn the required model.  This model is computed through an approximation to the Koopman Operator.  Once the model is learned, you can run the full system which shares control between the user and automation using Maxwell's Demon Algorithm (MDA).  A step-by-step set of instructions for how to run each part of the system can be found in the next section.  NOTE: When running the system, all user data and computed model files are stored in /koopman_operator_model_learning/data/ .


### Running the System
1. **Connect PS3 Controller**: To connect your PS3 joystick to your computer, open a terminal and run
```Shell
  sudo sixad -s
  ```
2. **Collect Observations**: To run the lunar lander simulator under direct control from a user (and collect observation data), open a terminal and run
```Shell
  roslaunch koopman_operator_model_learning collect_observations.launch
  ```
  When you have finished collecting data (only a small amount is necessary), you can press the space bar to shutdown the simulator.  The data will automatically be saved.
3. **Computing the Koopman**: To compute the Koopman operator from the collected data, open a terminal and run
```Shell
  roslaunch koopman_operator_model_learning compute_koopman.launch
  ```
4. **Run Experiment with shared control**: To run the lunar lander simulator with under the joint control paradigm, open a terminal and run
```Shell
  roslaunch koopman_operator_model_learning shared_control.launch
  ```

### Requirements
Software
1. ROS. Tested with ROS Indigo and Ubuntu 14.04
2. Python.  Tested with Python 2.7)
3. OpenAI gym.  We use a modified version of the basic OpenAI gym implementation.  In particular, we developed a Continuous Lunar Lander environment that integrates with ROS and allows user interaction.  You can pull the modified version of their code from [our fork](https://github.com/asbroad/gym).
4. scipy.
5. scikit-learn.scikit-learn is only used to store the data. It is easy to remove, and replace, this dependency if you should choose.

Hardware
1. PS3 Joystick.  Or a similar input device that can provide two dimensional contrinous control signals.

### Citing
If you find this code useful in your research, please consider citing:
```Shell
@inproceedings{broad2017learning,
    Author = {Alexander Broad, Todd Murphey and Brenna Argall},
    Title = {Learning Models for Shared Control of Human-Machine Systems with Unknown Dynamics},
    Booktitle = {Robotics: Science and Systems (RSS)},
    Year = {2017}
}
  ```

### License

The code in this repository is released under the MIT License (refer to the LICENSE file for details).


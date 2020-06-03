# Continuous Control


## Problem statement 
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. 
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. 

The task is episodic, and in order to solve the environment, the 
agent must get an average score of 30.0 over 100 consecutive episodes.

## Files
- `train.py`: Main script used train the agent 
- `agent.py`: Create the DDPG agent
- `model.py`: Contains the actor and critic networks
- `ExperienceReplay.py`: Random experience replay class
- `test.py`: Used to test the trained agent
- `report.pdf`: Report for the project

## Dependencies
To be able to run this code, you will need an environment with Python 3 and 
the dependencies are listed in the `requirements.txt` file so that you can install them
using the following command: 
```
pip install requirements.txt
``` 

Furthermore, you need to download the environment from one of the links below. You need only to select
the environment that matches your operating system:
- Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Running
- Clone the repository.
- Install the requirements.
- Run `test.py` to visualize your agent.

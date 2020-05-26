# Navigation


## Problem statement 
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided 
for collecting a blue banana. Thus, the goal of the agent is to collect 
as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions, and contains the agent's velocity, along
with ray-based perception of objects around the agent's forward
direction. Given this information, the agent has to learn how to best select 
actions. 
Four discrete actions are available, corresponding to: 
- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right  

The task is episodic, and in order to solve the environment, the 
agent must get an average score of +13 over 100 consecutive episodes.

## Files
- `train.py`: Main script used train the agent 
- `agent.py`: Create the DDQN agent
- `model.py`: Contains the Q-network responsible for prediction
- `ExperienceReplay.py`: Random experience replay class
- `PrioritizedReplay.py`: Prioritized experience replay class
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
- Linux : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- MAC OSX : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows : [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

## Running
- Run `train.py` to train the agent.
- The trained model `BananaBrain.pth` will be saved in `models`folder.
- Run `test.py` to visualize your agent.
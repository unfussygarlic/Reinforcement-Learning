from unityagents import UnityEnvironment
from collections import deque
import numpy as np
from agent import DQAgent
import gym
import torch
import matplotlib.pyplot as plt

def trainer(env,env_name,state_size,action_size,episodes = 3000,time_steps = 1000):
    score_res = []
    scores = deque(maxlen = 100)
    agent = DQAgent(state_size,action_size)
    for episode in range(episodes):
        score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        for i in range(time_steps):            
            action = int(agent.act(state))
            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            agent.step((state,action,reward,next_state,done))
            
            state = next_state
            score += reward
            
            if done:
                break
        scores.append(score)
        score_res.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\t eps: {:.5f}'.format(episode+1, np.mean(scores),agent.eps), end="")
        if (episode+1) %100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\t eps: {:.5f}'.format(episode+1, np.mean(scores),agent.eps))
        if np.mean(scores) >= 13.0:
            print(f'\nEnvironment solved in episode {episode+1} with the score of {np.mean(scores)}')
            torch.save(agent.local_network.state_dict(),f'{env_name}.pth')
            break
    return score_res,agent

def plot_scores(scores):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.title('Score (Rewards)')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.grid(True)      
    plt.savefig('images/score.png')
    plt.close()

if __name__ == '__main__':
    env = UnityEnvironment(file_name="data/Banana_Windows_x86_64/Banana.exe")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    score,agent = trainer(env,brain_name,37,4)
    plot_scores(score)
    env.close()
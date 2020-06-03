from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from agent import ddpgagent

def trainer(env,brain_name,state_size,action_size,num_agents,episodes = 3000,time_steps = 1000):
    score_res = []
    scores = deque(maxlen = 100)
    agent = ddpgagent(state_size,action_size,num_agents)
    eps = 1.0
    decay = 0.99995
    eps_min = 0.05
    for episode in range(episodes):
        score = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()

        for i in range(time_steps):          
            action = agent.act(state,eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done

            agent.step((state,action,reward,next_state,done))
            
            state = next_state
            score += reward

            if np.any(done):
                break
        
        eps = max(eps*decay,eps_min)
        scores.append(np.mean(score))
        score_res.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}\tEps: {}\tAction: {}'.format(episode+1, np.mean(scores),eps,agent.action_type), end="")
        if (episode+1) %100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode+1, np.mean(scores)))
        if np.mean(scores) > 30.0:
            print(f'\nEnvironment solved in episode {episode+1} with the score of {np.mean(scores)}')
            torch.save(agent.local_actor.state_dict(),f'models/actor.pth')
            torch.save(agent.local_critic.state_dict(),f'models/critic.pth')
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
    env = UnityEnvironment(file_name='data/20_Reacher_Windows_x86_64/Reacher.exe', base_port=63457)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode = True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]

    action_size = brain.vector_action_space_size

    score,agent = trainer(env,brain_name,state_size,action_size,num_agents)
    plot_scores(score)
    env.close()
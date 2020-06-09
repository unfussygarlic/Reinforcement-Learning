from unityagents import UnityEnvironment
import numpy as np
from agent import agent as Agent
from collections import deque
import pickle
import matplotlib.pyplot as plt

def trainer(state_size,action_size,num_agents,goal = 1.0,episodes = 5000,timesteps = 1000000):
    agent = Agent(state_size,action_size,num_agents)
    score_window = deque(maxlen = 100)
    scores = []
    max_score_window = deque(maxlen = 100)
    max_scores = []
    for episode in range(episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        local_score = np.zeros(num_agents)
        for i in range(timesteps):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            rewards = env_info.rewards
            next_states = env_info.vector_observations
            dones = env_info.local_done

            agent.step((states,actions,rewards,next_states,dones))
            states = next_states

            local_score += rewards

            if np.any(dones):
                break
        
        max_score_window.append(np.max(local_score))
        score_window.append(np.mean(local_score))
        scores.append(np.mean(local_score))
        max_scores.append(np.max(local_score))

        print('\rEpisode {}\tEpisode Score: {}'.format(episode+1,np.mean(max_score_window)), end="")

        if (episode+1) %100 == 0:
            print('\rEpisode {}\tEpisode Score:{}'.format(episode+1,np.mean(max_score_window)))
        
        if (episode+1) %1000 == 0:
            agent.save('models',episode)

        if np.mean(max_score_window) >= goal:
            print(f'\nEnvironment solved in episode {episode+1} with the score of {np.mean(max_score_window)}')
            agent.save('models',episode)
            break
    return scores,max_scores

def plot_scores(scores,name):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.title('Score (Rewards)')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.grid(True)      
    plt.savefig('images/{}.png'.format(name))
    plt.close()

def save_scores(scores,name):
    with open(f"{name}.txt", "wb") as s:
        pickle.dump(scores, s)

if __name__ == '__main__':
    env = UnityEnvironment(file_name = 'data/Tennis/Tennis.exe')

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset()[brain_name]

    num_agents = len(env_info.agents)
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size

    scores,max_scores = trainer(state_size,action_size,num_agents)
    
    save_scores(scores,'scores')
    save_scores(max_scores,'max_scores')

    plot_scores(scores,'scores')
    plot_scores(max_scores,'max_scores')
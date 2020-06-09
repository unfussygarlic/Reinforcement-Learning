from unityagents import UnityEnvironment
from agent import agent as Agent
import numpy as np
from time import sleep

def visualize(env, brain_name, agent, e = 962):
    mean_scores = []
    agent.load(f'models/sac_actor_e{e}.pth',f'models/td3_actor_e{e}.pth')
    env_info = env.reset(train_mode=True)[brain_name]    
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)

    while True:
        actions = agent.act(states)    
        
        env_info = env.step(actions)[brain_name]

        next_states = env_info.vector_observations         
        rewards = env_info.rewards                        
        dones = env_info.local_done                                   
        scores += rewards                                 
        states = next_states
        
        # sleep(0.005)
        if np.any(dones):                                 
            break

    print(f"\nFinal Mean Score : {np.mean(scores)}")
    print(f"Final Max Score : {np.max(scores)}")

    return np.mean(scores),np.max(scores)
    

if __name__ == '__main__':
    env = UnityEnvironment(file_name = 'data/Tennis/Tennis.exe')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset()[brain_name]

    num_agents = len(env_info.agents)
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size

    agent = Agent(state_size,action_size,num_agents)
    mean_score,max_score = visualize(env,brain_name,agent)
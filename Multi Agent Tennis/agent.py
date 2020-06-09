import torch
from sac import agent as SACAgent
from td3 import agent as TD3Agent
from ExperienceReplay import ReplayBuffer
import numpy as np
import glob

class agent(object):

    def __init__(self,state_size,action_size,num_agents,mode = 'COLAB',lr = 3e-4,seed = 1234):
        self.batch_size = 64

        agent1 = SACAgent(state_size,action_size,num_agents,lr,seed)
        agent2 = TD3Agent(state_size,action_size,num_agents,lr,seed)

        self.agent_names = ['sac','td3']

        self.agents = [agent1,agent2]

        self.memory = ReplayBuffer(int(1e6),self.batch_size)
        self.num_agents = num_agents
        self.mode = mode
    
    def act(self,states):
        actions = []
        for i in range(self.num_agents):
            actions.append(self.agents[i].act(states[i]))
        return np.array(actions)
    
    def step(self,observations):
        states,actions,rewards,next_states,dones = observations

        if self.mode == 'COMPETE':
            for i in range(self.num_agents):
                self.agents[i].update((states[i],actions[i],rewards[i],next_states[i],dones[i]))

        elif self.mode == 'COLAB':
            for i in range(self.num_agents):
                self.memory.add((states[i],actions[i],rewards[i],next_states[i],dones[i]))

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                for i in range(self.num_agents):
                    self.agents[i].train(experiences)
    
    def save(self,path,episode):
        for i in range(self.num_agents):
            save_path = "{}/{}_actor_e{}.pth".format(path,self.agent_names[i],episode)
            torch.save(self.agents[i].actor.state_dict(),save_path)
            save_path = "{}/{}_critic_e{}.pth".format(path,self.agent_names[i],episode)
            torch.save(self.agents[i].critic.state_dict(),save_path)
    
    def load(self,pathA,pathB):
        paths = [pathA,pathB]
        for i in range(self.num_agents):
            self.agents[i].actor.load_state_dict(torch.load(paths[i]))
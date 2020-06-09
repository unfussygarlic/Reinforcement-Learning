from collections import deque,namedtuple
import random
import torch
import numpy as np

class ReplayBuffer:

    def __init__(self,buffer_size = 10000,batch_size = 32):
        self.memory = deque(maxlen = buffer_size)
        self.priority = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience",field_names = ['state','action','reward','next_state','done'])
        self.batch_size = batch_size
    
    def add(self,experience):
        state,action,reward,next_state,done = experience
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        self.priority.append(max(self.priority,default = 1))
    
    def set_priorities(self,alpha):
        sample = np.array(self.priority)**alpha
        sample = sample/sum(self.priority)
        return sample
    
    def get_weights(self,priorities,beta):
        sample = (1/len(self.memory)) * (1/priorities)
        sample_norm = sample/max(sample)
        # sample = sample**(-beta)
        return sample_norm
    
    def sample(self,alpha = 0.98,beta = 0.6):
        size = min(len(self.memory),self.batch_size)
        pr = self.set_priorities(alpha = alpha)
        indices = random.choices(range(len(self.memory)),k = size,weights = pr)
        
        priorities = [self.priority[i] for i in indices] 
        weights = self.get_weights(np.array(priorities),beta = beta)

        states = torch.from_numpy(np.vstack([self.memory[i].state for i in indices if self.memory[i] is not None])).float()
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in indices if self.memory[i] is not None])).long()
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in indices if self.memory[i] is not None])).float()
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in indices if self.memory[i] is not None])).float()
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in indices if self.memory[i] is not None]).astype(np.uint8)).float()
        weights = torch.from_numpy(np.vstack([i for i in weights])).float()

        return (states,actions,rewards,next_states,dones),weights,indices
    
    def update_priority(self,indices,errors,offset = 0.1):
        for i,e in zip(indices,errors):
            self.priority[i] = abs(e) + offset
    
    def __len__(self):
        return len(self.memory)
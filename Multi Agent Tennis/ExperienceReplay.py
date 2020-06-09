from collections import namedtuple,deque
import random
import torch
import numpy as np

#Setting up device for torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:

    def __init__(self,buffer_size,batch_size):
        """ 
        Replay buffer initialization

        Arguments:
        buffer_size : Maximun length of the buffer
        batch_size : Sampling size for each iteration
        """
        self.batch_size = batch_size

        #Deque is used to initialize memory which holds experiences
        self.memory = deque(maxlen = buffer_size)

        #Named tuple used for ease of extraction
        self.experience = namedtuple('Experience',field_names = ['state','action','reward','next_state','done'])
    
    def add(self,experience):
        """
        Experiences are added on each time step

        Argument:
        experience : Tuple containing experience passed by agent
        """        
        state,action,reward,next_state,done = experience
        e = self.experience(state,action,reward,next_state,done)

        #Experience (named tuple) is appended to the memory
        self.memory.append(e)
    
    def sample(self):
        """ 
        Return random batch experiences from the memory.
        Indices of experiences are sampled from memory randomly.
        Experiences are stacked using np.vstack and converted into
        tensor for pytorch training.
        """
        batch_size = min(self.batch_size,len(self.memory))
        indices = random.choices(range(len(self.memory)),k = batch_size)

        states = torch.from_numpy(np.vstack([self.memory[i].state for i in indices if self.memory[i] is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in indices if self.memory[i] is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in indices if self.memory[i] is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in indices if self.memory[i] is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in indices if self.memory[i] is not None]).astype(np.uint8)).float().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        """
        Returns current length of the memory (buffer)
        """
        return len(self.memory)
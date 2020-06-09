from models import TD3Policy,Critic
from ExperienceReplay import ReplayBuffer
import torch
import torch.optim as opt
import torch.nn.functional as F
import numpy as np

class agent(object):

    def __init__(self,state_size,action_size,num_agents,lr,seed):
        """ 
        Twin Delayed DDPG agent.

        Arguments:
        state_size : Size of the state from environment.
        action_size : Size of the action taken by the agent
        num_agents : Total number of agents
        lr : Common learning rate for both agents
        seed : Seed value for reproducibility
        """
        #Initialization of local and target actor networks
        self.actor = TD3Policy(state_size,action_size,seed)
        self.actor_opt = opt.Adam(self.actor.parameters(),lr = lr)
        self.actor_target = TD3Policy(state_size,action_size,seed)

        #Initialization of local and target critic networks
        self.critic = Critic(state_size,action_size,seed)
        self.critic_opt = opt.Adam(self.critic.parameters(),lr = lr)
        self.critic_target = Critic(state_size,action_size,seed)

        #Agent hyper parameters
        self.num_agents = num_agents
        self.policy_update = 2
        self.step = 0
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.gamma = 0.998
        self.TAU = 0.005
        self.batch_size = 64

        #Replay buffer for 'COMPETE' mode
        self.memory = ReplayBuffer(int(1e6),self.batch_size)

        #Hard updation of target network
        self.hard_update(self.critic,self.critic_target)
    
    def act(self,state,add_noise = True):
        """
        Returns action for the given state

        Argument:
        state : State vector containing state variables
        """
        state = torch.from_numpy(state).float()
        self.actor.eval()
        with torch.no_grad():
            actions =   self.actor(state).cpu().data.numpy()
        self.actor.train()
        return actions
    
    def update(self,experience):
        """
        Updates the agent's replay buffer and trains the agent
        in 'COMPETE' mode.

        Arguments:
        experience : Tuple containing current experience
        """
        self.memory.add(experience)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.train(experiences)    
    
    def train(self,observations):
        """
        Trains the agent using the experiences

        Arguments:
        experience : Tuple containing current experience
        """
        states,actions,rewards,next_states,dones = observations
        #Step is updated at each timestep for policy updation
        self.step = (self.step+1) % self.policy_update

        """
        Random noise is added to the Q_target. This encourages the 
        agent to explore more. Minimum of the clipped Q values is
        used to reduce the overestimation behaviour of DDPG
        """
        with torch.no_grad():
            noise = (torch.randn_like(actions)*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1,1)
            Q1_target,Q2_target = self.critic_target(next_states,next_actions)
            Q_target = torch.min(Q1_target,Q2_target)
            Q_target = rewards + (self.gamma*Q_target*(1-dones))
        
        Q1_expected,Q2_expected = self.critic(states,actions)

        critic_loss = F.mse_loss(Q1_expected,Q_target) + F.mse_loss(Q2_expected,Q_target)

        #Updating local critic network
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_opt.step()

        """
        Policy is updated every self.policy_update timesteps.
        ie., if the policy update is set to 2, Policy network is updated 
        every two times the updation of critic network.
        This stabilizes the learning and overestimation.
        """
        if self.step == 0:
            expected_actions = self.actor(states)
            actor_loss = -self.critic.Q1(states,expected_actions).mean()

            #Updating local 
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            #Soft update of actor and critic target network using TAU
            self.soft_update(self.actor,self.actor_target)
            self.soft_update(self.critic,self.critic_target)
    
    def soft_update(self,local,target):
        """ 
        Soft update of target network parameters using TAU
        """
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(self.TAU*l.data + (1-self.TAU)*t.data)
    
    def hard_update(self,local,target):
        """
        Hard update which copies local network parameters to target network
        """
        for l,t in zip(local.parameters(),target.parameters()):
            t.data.copy_(l.data)
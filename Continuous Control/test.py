from unityagents import UnityEnvironment
import torch
from model import Actor as network
import numpy as np
from time import sleep

def act(state,checkpoint,num_agents):
    actions = []
    for i in range(num_agents):
        state_tensor = torch.from_numpy(state[i]).float()
        model = network(33,4)
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        with torch.no_grad():
            action = model(state_tensor).data.numpy()
        actions.append(action)
    return actions

if __name__ == '__main__':
    env = UnityEnvironment(file_name="data/20_Reacher_Windows_x86_64/Reacher.exe")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    num_agents = len(env_info.agents)
    done = False
    score = np.zeros(num_agents)
    t = 0
    while True:
        action = act(state,'models/actor.pth',num_agents)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        done = env_info.local_done
        state = next_state
        score += env_info.rewards
        t+=1
        print("\rTimestep: {}\tScore: {:.2f}".format(t,np.mean(score)),end = "")
        if np.mean(score) > 30.0:
            break
    env.close()
    print(f"\nFinal Score : {np.mean(score)}")
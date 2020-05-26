from unityagents import UnityEnvironment
import torch
from model import network
import numpy as np
from time import sleep

def act(state,checkpoint):
    state = torch.from_numpy(state).float().unsqueeze(0)
    model = network(37,4)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    with torch.no_grad():
        action = model(state)
    action = np.argmax(action.numpy())
    return action

if __name__ == '__main__':
    env = UnityEnvironment(file_name="data/Banana_Windows_x86_64/Banana.exe")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    done = False
    score = 0
    while not done:
        action = int(act(state,'models/BananaBrain.pth'))
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        state = next_state
        score += env_info.rewards[0]
        sleep(0.05)
    env.close()
    print(f"Score : {score}")
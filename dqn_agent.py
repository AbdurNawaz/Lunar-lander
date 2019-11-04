import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.optim as optim
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-5
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Agent():

    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.t_step = 0


    def step(self, state, action, reward, next_state, done):
        #This function saves the experience in the replay memory

        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1)%UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps):
        #This function returns actions for given states as per the current policy

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_values))

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        Q_targets = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        Q_targets = rewards + (gamma*Q_targets*(1 - dones))

        Q_expected =  self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.eval()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_layer, target_layer, tau):

        for target_param, local_param in zip(target_model.parameters(), local_layer.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


    

#region Imports
import random
import numpy as np
from ReplayBuffer import ReplayBuffer
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

#endregion

#region hyperparameters

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

#endregion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #Q-NETWORK
        self.qnetwork_local = QNetwork(state_size,action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size,action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        #REPLAY MEMORY
        self.memory = ReplayBuffer(action_size,BUFFER_SIZE,BATCH_SIZE,seed)

        #Initialize time step ( for updating every UPDATE_EVERY steps )
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        #Save experience in replay memory
        self.memory.add(state,action, reward,next_state,done)

        #Learn every UPDATE_EVERY time steps ( in our case, learn every 4 time steps )
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 :
            #If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences,GAMMA)
    def act(self, state, eps=0.) :

        """Returns actions for given state as per current policy.

        Params
        =======
        state (array_like): current state
        eps (float): epsilon, for epsilon-greedy action selection
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor

        """



        states, actions, rewards, next_states, dones = experiences


        old_val = self.qnetwork_local(states).gather(-1, actions)
        with torch.no_grad():
            next_actions = self.qnetwork_local(next_states).argmax(-1, keepdim=True)
            maxQ = self.qnetwork_target(next_states).gather(-1, next_actions)
            target = rewards + GAMMA * maxQ * (1 - dones)

        #Compute loss
        loss = F.mse_loss(old_val, target)

        #Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target ,TAU)

    def soft_update(self, local_model, target_model, tau):

        """Soft update model parameters.
                θ_target = τ*θ_local + (1 - τ)*θ_target
                Params
                ======
                    local_model (PyTorch model): weights will be copied from
                    target_model (PyTorch model): weights will be copied to
                    tau (float): interpolation parameter
                """

        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


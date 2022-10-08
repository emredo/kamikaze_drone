import random
import torch
import torch.nn as nn
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden1, hidden2):
        super(ActorCritic, self).__init__()
        
        self.base = nn.Sequential(
            nn.Linear(num_inputs, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )

        self.critic = nn.Sequential(
            nn.Linear(hidden2, 1)
        )

        self.mu_net = nn.Sequential(
            nn.Linear(hidden2, num_outputs),
            nn.Tanh()
        )

        self.variance = nn.Sequential(
            nn.Linear(hidden2, num_outputs),
            nn.Softplus()
        )

    def forward(self, x):
        base_out = self.base(x)
        value = self.critic(base_out)
        mu = self.mu_net(base_out)
        var = self.variance(base_out)
        dist = torch.distributions.Normal(mu, var**(1/2))
        return dist, value


class ReplayMemory:
    def __init__(self,batch_size,max_len):
        self.batch_size = batch_size
        self.max_len = max_len
        self.memory = deque(maxlen=self.max_len)

    def __len__(self):
        return len(self.memory)

    def add_data(self,state,action,log_prob, reward, value, done):
        self.memory.append((state,action,log_prob, reward, value, done))

    def get_memory(self):
        arr = np.array(self.memory, dtype=object)
        states = arr[:,0]
        actions = arr[:,1]
        log_probs = arr[:, 2]
        rewards = arr[:, 3]
        values = arr[:,4]
        dones = arr[:, 5]
        return np.vstack(states),np.vstack(actions),np.vstack(log_probs), \
               np.vstack(rewards), np.vstack(values), np.vstack(dones)

    def generate_batches(self,states,actions,log_probs,returns,values):
        indexes = list(np.arange(len(states)))
        random.shuffle(indexes)
        batch_indexes = np.array_split(indexes,self.max_len//self.batch_size)

        batches = []
        for batch_index in batch_indexes:
            batch = []
            for index in batch_index:
                batch.append((states[index],actions[index],log_probs[index],returns[index],values[index]))
            batches.append(batch)
        return batches


class Updater:
    def __init__(self,net, clip_param, critic_discount, entropy_beta, learning_rate=0.001):
        self.net = net
        self.clip_param = clip_param
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=learning_rate)

    def update(self,states,actions,old_log_probs,returns, advantage):
        dist, new_value = self.net(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)
        ratio = new_log_probs.exp() / old_log_probs.exp()

        surrogate1 = ratio*advantage
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

        actor_loss = -torch.min(surrogate1,surrogate2).mean()
        critic_loss = (returns - new_value).pow(2).mean()

        total_loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


def gae_calculator(next_value, rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - int(masks[step])) - values[step]
        gae = delta + gamma * lam * (1-int(masks[step])) * gae
        returns.insert(0, gae + values[step])
    return returns


def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def visualize(file_path):
    file = open(file_path,"r")
    lines = file.readlines()
    plotting_indexes = range(0,len(lines),10)

    rewards = []
    avg_rewards = []
    for line in lines:
        reward = line.split(" ")[1][:-1]
        rewards.append(float(reward))
    for index, reward in enumerate(rewards):
        if len(rewards) - index > 10:
            avg_10_reward = np.array(rewards[index:index+10],dtype=float).mean()
            avg_rewards.append(avg_10_reward)

    rewards_arr = np.array(rewards,dtype=float)
    avg_arr = np.array(avg_rewards,dtype=float)
    plt.plot(plotting_indexes, rewards_arr[plotting_indexes])
    plt.plot(plotting_indexes[:-1], avg_arr[plotting_indexes[:-1]],c="red")
    plt.legend(["epoch_reward","10_epochs_average_reward"])
    plt.show()

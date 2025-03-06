import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import rldurham as rld
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Define utility functions
def evaluate_policy(env, agent, turns=3):
    scores = 0
    for j in range(turns):
        s, _ = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.sample_action(s, deterministic=True)
            s_next, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            s = s_next
            scores += r
    return scores/turns

# Actor and Critic network definitions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, net_width):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

# ReplayBuffer for storing transitions
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.done[ind])
        )

# Agent implementation based on template
class Agent(torch.nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        # TD3 hyperparameters
        self.gamma = 0.99
        self.net_width = 256
        self.a_lr = 1e-4
        self.c_lr = 1e-4
        self.batch_size = 256
        self.delay_freq = 2
        self.explore_noise = 0.1
        self.explore_noise_decay = 0.998
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Will be initialized in setup_networks
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.replay_buffer = None
        
        self.max_action = 1.0  # Default for Walker environment
        self.state_dim = None
        self.action_dim = None
        self.total_it = 0
        
        # Flag to track if networks are set up
        self.is_initialized = False
        self.training_started = False

    def setup_networks(self, state_dim, action_dim, max_action=1.0):
        """Initialize networks with correct dimensions"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.actor = Actor(state_dim, action_dim, max_action, self.net_width).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, self.net_width).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.critic = Critic(state_dim, action_dim, self.net_width).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.net_width).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.is_initialized = True

    def sample_action(self, s, deterministic=False):
        """Select action according to policy with/without noise"""
        if not self.is_initialized:
            # Initialize networks if this is the first call
            if isinstance(s, np.ndarray):
                state_dim = s.shape[0]
            else:  # If tensor
                state_dim = s.size(0)
            self.setup_networks(state_dim, act_dim)
        
        with torch.no_grad():
            if isinstance(s, np.ndarray):
                state = torch.FloatTensor(s.reshape(1, -1)).to(self.device)
            else:  # If already tensor
                state = s.reshape(1, -1).to(self.device)
                
            action = self.actor(state).cpu().data.numpy().flatten()
            
            if not deterministic:
                noise = np.random.normal(0, self.explore_noise, size=self.action_dim)
                action = action + noise
                action = np.clip(action, -self.max_action, self.max_action)
                
        return action

    def put_data(self, action, observation, reward, next_observation=None, done=False):
        """Add a transition to the replay buffer"""
        if not next_observation:  # For compatibility with template interface
            # In this case, 'observation' is actually the next observation
            next_observation = observation
            observation = self.last_observation
            
        if not hasattr(self, 'last_observation'):
            self.last_observation = observation
            return
        
        if next_observation is None:
            next_observation = observation
            observation = self.last_observation
            done = False  # No done signal in template format

        # Store the transition in the replay buffer
        self.replay_buffer.add(self.last_observation, action, reward, observation, done)
        self.last_observation = observation

    def train(self):
        """Train the agent (one iteration of TD3 training)"""
        self.total_it += 1
        self.training_started = True
        
        # Sample replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * self.policy_noise * self.max_action).clamp(
            -self.noise_clip * self.max_action, self.noise_clip * self.max_action)
        next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.delay_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)
                
            # Decay exploration noise
            self.explore_noise *= self.explore_noise_decay

    def can_train(self):
        """Check if there are enough samples to start training"""
        return self.replay_buffer.size >= self.batch_size

    def save(self, filename):
        """Save model parameters"""
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        """Load model parameters"""
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

# Main script that sets up and runs the training loop
env = rld.make("rldurham/Walker", render_mode="rgb_array")
# env = rld.make("rldurham/Walker", render_mode="rgb_array", hardcore=True) # only attempt this after solving non-hardcore

# get statistics, logs, and videos
env = rld.Recorder(
    env,
    smoothing=10,                       # track rolling averages
    video=True,                         # enable recording videos
    video_folder="videos",              # folder for videos
    video_prefix="xxxx00-agent-video",  # replace xxxx00 with your username
    logs=True,                          # keep logs
)

# training on CPU recommended
rld.check_device()

# environment info
discrete_act, discrete_obs, act_dim, obs_dim = rld.env_info(env, print_out=True)

# render start image
env.reset(seed=42)
rld.render(env)

# seed everything for verification
seed, observation, info = rld.seed_everything(42, env)

# initialize agent
agent = Agent()
max_episodes = 1000
max_timesteps = 2000
min_buffer_size = 3000  # Minimum transitions before training
update_after = 5000     # Start training after this many steps
update_every = 50       # Train this many times each episode

# track statistics for plotting
tracker = rld.InfoTracker()

# switch video recording off (only switch on every x episodes as this is slow)
env.video = False

# Total environment steps counter
total_steps = 0

# training procedure
for episode in range(max_episodes):
    
    # recording statistics and video can be switched on and off
    env.info = episode % 50 == 0   # track every 10 episodes
    env.video = episode % 50 == 0  # record videos every 10 episodes
    
    # reset for new episode
    observation, info = env.reset()
    
    # Store the first observation
    agent.last_observation = observation
    episode_reward = 0
    
    # run episode
    for t in range(max_timesteps):
        
        # select the agent action
        action = agent.sample_action(observation, deterministic=False)
        
        # take action in the environment
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # add to agent memory
        agent.put_data(action, next_observation, reward, None, terminated or truncated)
        
        # update observation
        observation = next_observation
        episode_reward += reward
        total_steps += 1
        
        # Start training when enough samples are collected
        if total_steps >= min_buffer_size and agent.replay_buffer.size >= agent.batch_size:
            if total_steps % update_every == 0:
                for _ in range(update_every):
                    agent.train()
        
        # check whether done
        done = terminated or truncated
        
        # stop episode
        if done:
            break
            
    # Print episode info
    #if (episode + 1) % 10 == 0:
    print(f"Episode {episode+1}, Reward: {episode_reward}, Steps: {total_steps}")
    
    # Decay exploration noise
    if agent.training_started:
        agent.explore_noise *= agent.explore_noise_decay
            
    # track and plot statistics
    tracker.track(info)
    #if (episode + 1) % 100 == 0:
        #tracker.plot(r_mean_=True, r_std_=True, r_sum=dict(linestyle=':', marker='x'))

# don't forget to close environment (e.g. triggers last video save)
env.close()

# write log file (for coursework)
env.write_log(folder="logs", file="xxxx00-agent-log.txt")  # replace xxxx00 with your username

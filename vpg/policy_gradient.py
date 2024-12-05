# vanila policy gradient descent with reward to go and BASELINE
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym_pygame
import numpy as np
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
from utils import record_video

# hyperparamebers to tune
learning_rate =  1e-4
gamma = 0.9

debug = False

# File path to the model
policy_model_path = "policy_model.pth"
value_model_path = "value_model.pth"
video_fps = 30
# video_path = 'replay.mp4'


def load_model(model, model_file_path):
  # Check if the model file exists
    if os.path.exists(model_file_path):
        print(f"Loading model from {model_file_path}...")
        # Load the saved model state
        model.load_state_dict(torch.load(model_file_path))
    else:
        print(f"No saved model found at {model_file_path}. Creating a new model...")
        
def save_model(model, model_file_path):
    # Save model
    torch.save(model.state_dict(), model_file_path)

def disable_random():
    # disable all random ness
    # Set the random seed for Python's random module
    random.seed(0)
    # Set the random seed for NumPy (if you're using it)
    np.random.seed(0)
    # Set the random seed for PyTorch (CPU and GPU)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

def print_trajectories(batch_traj):
    #debug if action is random enough
    if debug:
        actions = []
        for st, at, reward, st_1, done in batch_traj:
            actions.append(at)
        print(f'actions: {actions}')

def print_rewards(traj_rewards):
    #debug if the random b actually yield difference in rewards
    if debug:
        print(f'traj rewards: {traj_rewards}')

class Policy(nn.Module):
    # (TODO) MLP/Policy can be reconfigured to contain multiple layers
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 2 * dim_hidden),
            nn.ReLU(),
            nn.Linear(2 * dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output)
        )
    # return logits of size(dim_output)
    def forward(self, x):
        return self.mlp(x)
    
    # action(st), H(st) -> at based on probability distribution
    def take_action(self, s):
        logits = self.mlp(s)
        probs = torch.softmax(logits, dim=-1)
        categorical_dist = torch.distributions.Categorical(probs = probs)
        action = categorical_dist.sample()
        return action.item()

    def get_logp(self, obs, acts):
        logits = self.mlp(obs)
        categorical_dists = torch.distributions.Categorical(logits = logits)
        return categorical_dists.log_prob(acts)
    
class Value(nn.Module):
    # (TODO) MLP/Policy can be reconfigured to contain multiple layers
    def __init__(self, dim_input, dim_hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 2 * dim_hidden),
            nn.ReLU(),
            nn.Linear(2 * dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1)
        )
        
    def forward(self, x):
        return self.mlp(x)


# return loss, avg_reward, avg_lens for this epoch
def train_epoch(env, policy, value, num_batch, policy_opt, value_opt):
    # return [(st, at, reward, st+1, done)]
    def generate_trajectory(env, policy, num_batch):
        batch_traj = []
        for i in range(num_batch):
            st = env.reset()
            done = False
            while not done:
                action = policy.take_action(torch.tensor(st))
                st_1, reward, done, _ = env.step(action)
                batch_traj.append((st, action, reward, st_1, done))
                st = st_1
        return batch_traj
    
    # return reward to go
    def compute_reward_to_go(batch_traj):
        rtgs = []
        curr_traj = []
        for st, at, reward, st_1, done in batch_traj:
            curr_traj.append((st, at, reward, st_1, done))
            if done:
                # append rewards to gos from current trajectory to rtgs
                rtgs.extend(get_rtgs_from_current_traj(curr_traj))
                curr_traj = []
        # normalize reward
        rtgs = (rtgs - np.mean(rtgs)) / (1e-9 + np.std(rtgs))
        return rtgs
                
    def get_rtgs_from_current_traj(curr_traj):
        n = len(curr_traj)
        rtgs = [0] * n
        for i in reversed(range(n)):
            st, at, reward, st_1, done = curr_traj[i]
            if i == n - 1:
                rtgs[i] = reward
            else:
                rtgs[i] = reward + gamma * rtgs[i + 1]
        return rtgs 
    
    # traj: [(st, at, reward, st+1, done)]
    def fit_value_evaluation(value, value_opt, batch_traj, batch_rtgs):
        predications = []
        # episode_in_traj = []
        for st, at, reward, st_1, done in batch_traj:
            v_st = value(torch.as_tensor(st))
            predications.append(v_st)
        value_opt.zero_grad()
        targets = torch.as_tensor(batch_rtgs, dtype=torch.float64)
        predications = torch.stack(predications).squeeze(-1)
        loss = nn.MSELoss()(predications, targets)
        loss.backward()
        value_opt.step()
        return loss.item()
        
    
    def fit_policy_evaluation(policy, policy_opt, batch_traj, value, batch_rtgs):
        batch_adv = [] # [tensor(torch.double)]
        batch_baseline = []
        batch_st = []
        batch_at = []
        # compute adv(st, at)
        with torch.no_grad():
            for st, at, reward, st_1, done in batch_traj:
                v_st = value(torch.as_tensor(st)).item()
                batch_baseline.append(v_st)
                batch_st.append(st)
                batch_at.append(at)
        batch_adv = torch.as_tensor(batch_rtgs) - torch.as_tensor(batch_baseline)
        logps = policy.get_logp(obs= torch.as_tensor(batch_st), acts = torch.as_tensor(batch_at))
        loss = -(logps * batch_adv).mean()
        policy_opt.zero_grad()
        loss.backward()
        policy_opt.step()
        return loss.item()
    
    # return mean(reward), len(reward), std(reward)
    def evaluate_policy(batch_traj):
        sampled_rewards = []
        traj_lens = []
        traj_reward = 0
        len = 0
        for st, at, reward, st_1, done in batch_traj:
            len += 1
            traj_reward += reward
            if done:
                sampled_rewards.append(traj_reward)
                traj_lens.append(len)
                traj_reward = 0
                len = 0
        return np.mean(sampled_rewards), np.std(sampled_rewards), np.mean(traj_lens)
                
        
    # start = timer()
    batch_traj = generate_trajectory(env, policy, num_batch)
    batch_rtgs = compute_reward_to_go(batch_traj)
    print_trajectories(batch_traj)
    print_rewards(batch_rtgs)
    v_loss = fit_value_evaluation(value = value, value_opt=value_opt, batch_traj=batch_traj, batch_rtgs=batch_rtgs)
    p_loss = fit_policy_evaluation(policy = policy, policy_opt = policy_opt, batch_traj = batch_traj, value = value, batch_rtgs=batch_rtgs)
    reward_mean, reward_std, len_mean = evaluate_policy(batch_traj)
    
    return p_loss, v_loss, reward_mean, len_mean, reward_std
def train():
    start = timer()
    env = gym.make("Pixelcopter-PLE-v0")
    # make the action deterministic
    dim_input = env.observation_space.shape[0]
    dim_output = env.action_space.n
    policy = Policy(dim_input, 32, dim_output).double()
    value = Value(dim_input, 32).double()
    # load_model(policy, policy_model_path)
    # load_model(value, value_model_path)
    policy_opt = torch.optim.Adam(policy.parameters(), lr = learning_rate)
    value_opt = torch.optim.Adam(value.parameters(), lr = learning_rate)

    num_epoch = 1000
    num_batch = 10
    rewards = []
    reward_std = []
    for i in range(num_epoch):
        p_loss, v_loss, reward, lens, reward_st = train_epoch(env, policy, value, num_batch, policy_opt, value_opt)
        if i % 10 == 0:
            print(f"{i}th iteration, p loss={p_loss}, v_loss={v_loss} rewards={reward}, std={reward_st} lens={lens}\n")
        # print(f"{i}th iteration, loss={loss}, rewards={reward}, std={reward_st} lens={lens}\n")

        rewards.append(reward)
        reward_std.append(reward_st)
    print(f'total time for traning: {timer() - start}')
    mean_minus_std = ((torch.as_tensor(rewards) - torch.as_tensor(reward_std))).numpy()
    plt.plot(list(range(num_epoch)), rewards, marker='o', linestyle='-', label='Rewards per Epoch')
    plt.plot(list(range(num_epoch)), reward_std, marker='s', linestyle='--',  color='red', label='std per Epoch')
    # plt.plot(list(range(num_epoch)), mean_minus_std, marker='s', linestyle='-',  color='green', label='mean - std per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.title('Rewards per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
    # save_model(policy, policy_model_path)
    # save_model(value, value_model_path)
    policy.eval()
    current_time = datetime.now()

    # Format the date and time as mm-dd-tt
    formatted_time = current_time.strftime("%m-%d-%H:%M")  # %H%M represents hours and minutes (24-hour format)

    # Append the formatted time to the string 'replay'
    video_path = f"replay-{formatted_time}.mp4"
    record_video(env, policy, video_path, video_fps)

if __name__ == "__main__":
    train()

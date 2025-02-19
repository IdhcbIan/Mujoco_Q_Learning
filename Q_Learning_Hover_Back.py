#------// Imports //------------------------------

import mujoco
from mujoco import viewer
import numpy as np
import time
import threading
import logging

import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

#------// Configuration and Logging //------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Initialize TensorBoard writer
writer = SummaryWriter('runs/Q_Learning_Hover')

#------// Setting Up NN   //------------------------------
# Set up device

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# Define the replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)  # Outputs raw Q-values

#------// Setting Up Drone Environment   //------------------------------
# Load the model from the XML file
model = mujoco.MjModel.from_xml_path('skydio_x2/scene.xml')

# Create a data object
data = mujoco.MjData(model)

# Initialize control inputs to zero
data.ctrl[:] = 0.0

# Define thrust levels
max_thrust = 5
return_thrust = 3
increment = 0.1  # Define the increment step

# ---- Desired Altitude ----
desired_z = 5.0  # Set your desired altitude

def reset_simulation():
    """Resets the simulation to its initial state."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0  # Reset control inputs
    print("Simulation has been reset.")

def thrust_control():
    """Controls the drone thrust using the neural network and resets simulation upon landing."""
    global data, model, start_time
    action_counts = {0:0, 1:0, 2:0, 3:0}  # Initialize action counts
    start_time = time.time()  # Start time for tracking flight duration
    
    while True:
        # Get current state
        state = get_state()
        state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
        with torch.no_grad():
            action = policy_net(state_tensor).max(1).indices.item()

        action_counts[action] += 1  # Count the action

        # Map action to thrust value
        # Example mapping: 0=decrease thrust, 1=maintain, 2=increase, 3=max thrust
        if action == 0:
            thrust_value = max(return_thrust - increment, 0)
        elif action == 1:
            thrust_value = return_thrust
        elif action == 2:
            thrust_value = return_thrust + increment
        elif action == 3:
            thrust_value = max_thrust
        else:
            thrust_value = return_thrust  # Default action

        data.ctrl[:] = [thrust_value] * model.nu
        mujoco.mj_step(model, data)

        # Get drone position
        drone_x = data.qpos[0]
        drone_y = data.qpos[1]
        drone_z = data.qpos[2]

        # Calculate current time in air
        current_flight_time = time.time() - start_time

        # Calculate reward
        reward = current_flight_time - (abs(drone_x) + abs(drone_y)) * 5
        
        print(f"Time: {current_flight_time:.2f}, Thrust: {thrust_value:.2f}, Position: x={drone_x:.2f}, y={drone_y:.2f}, z={drone_z:.2f}")
        print(f"Time in air: {current_flight_time:.2f}")
        print(f"Reward: {reward}")

        # Only check for resets after the first second
        if current_flight_time > 8.0:
            if drone_z < 0.01:
                start_time = time.time()  # Reset start time for tracking flight duration
                reset_simulation()        # Reset the simulation when drone is on the ground
                # Log action distribution
                logging.info(f"Action distribution in last episode: {action_counts}")
                action_counts = {0:0, 1:0, 2:0, 3:0}  # Reset counts for next episode

        # Only check for reward-based reset after the first second
        if current_flight_time > 2.0 and reward <= -20:
            start_time = time.time()
            reset_simulation()
            print("Resetting due to low reward")
            # Reset action counts on reward-based reset
            action_counts = {0:0, 1:0, 2:0, 3:0}

        time.sleep(0.01)

#------// Setting Up Hyperparameters   //------------------------------

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Define state space and actions
n_actions = 4  # Four possible actions: decrease, maintain, increase, max thrust
n_observations = 9  # [x, y, z positions, x, y, z velocities, roll, pitch, yaw]

# Initialize networks and optimizer
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# Function to get current state
def get_state():
    """Retrieves the current state of the drone."""
    state = np.array([
        data.qpos[0],  # x position
        data.qpos[1],  # y position
        data.qpos[2],  # z position
        data.qvel[0],  # x velocity
        data.qvel[1],  # y velocity
        data.qvel[2],  # z velocity
        data.qpos[3],  # roll
        data.qpos[4],  # pitch
        data.qpos[5]   # yaw
    ])
    # Example normalization (adjust based on actual ranges)
    state /= np.array([10, 10, 10, 10, 10, 10, np.pi, np.pi, np.pi])
    return state

steps_done = 0

#------// Model Tuning   //------------------------------

def select_action(state):
    """Selects an action using an epsilon-greedy policy."""
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        random_action = random.randint(0, n_actions - 1)
        return torch.tensor([[random_action]], device=device, dtype=torch.long)

def optimize_model():
    """Performs a single optimization step."""
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    # Log the loss to TensorBoard
    writer.add_scalar('Loss', loss.item(), global_step=steps_done)

def training_loop():
    """Runs the training loop for the DQN."""
    global start_time  # Declare start_time as global
    num_episodes = 600
    for i_episode in range(num_episodes):
        # Reset the Mujoco simulation
        reset_simulation()
        start_time = time.time()  # Track the start time of the episode
        state = torch.tensor(get_state(), dtype=torch.float32, device=device).unsqueeze(0)

        episode_reward = 0  # Initialize episode reward

        for t in count():
            action = select_action(state)
            # Map action to thrust value for simulation
            if action.item() == 0:
                thrust_value = max(return_thrust - increment, 0)
            elif action.item() == 1:
                thrust_value = return_thrust
            elif action.item() == 2:
                thrust_value = return_thrust + increment
            elif action.item() == 3:
                thrust_value = max_thrust
            else:
                thrust_value = return_thrust  # Default action

            data.ctrl[:] = [thrust_value] * model.nu
            mujoco.mj_step(model, data)

            # Get new state
            next_state = torch.tensor(get_state(), dtype=torch.float32, device=device).unsqueeze(0)

            # Get drone position
            drone_x = data.qpos[0]
            drone_y = data.qpos[1]
            drone_z = data.qpos[2]

            # Calculate current flight time
            current_flight_time = time.time() - start_time

            # Calculate reward
            desired_z = 5.0
            reward = current_flight_time - (abs(drone_x) + abs(drone_y) + abs(drone_z) - desired_z) * 5
            episode_reward += reward  # Accumulate rewards
            reward_tensor = torch.tensor([reward], device=device)

            # Check if episode is done
            done = (current_flight_time > 2.0 and drone_z < 0.01) or (current_flight_time > 2.0 and reward <= -20) or (current_flight_time > 2.0 and (abs(data.qpos[3]) > 0.5 or abs(data.qpos[4]) > 0.5 or abs(data.qpos[5]) > 0.5))
            
            if done:
                next_state = None
                reset_simulation()
                if reward <= -20:
                    print("Resetting due to low reward")

            # Store the transition in memory
            memory.push(state, action, next_state, reward_tensor)
            state = next_state if not done else torch.tensor(get_state(), dtype=torch.float32, device=device).unsqueeze(0)

            optimize_model()

            # Update the target network
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = (policy_net_state_dict[key] * TAU +
                                              target_net_state_dict[key] * (1 - TAU))
            target_net.load_state_dict(target_net_state_dict)

            if done:
                logging.info(f"Episode {i_episode + 1} finished after {t + 1} steps with reward: {episode_reward:.2f}")
                break
    print('Training Complete')
    writer.close()

#------// Starting Threads   //------------------------------

# Initialize start_time before starting threads
start_time = time.time()

# Start the training and control threads
training_thread = threading.Thread(target=training_loop, daemon=True)
training_thread.start()

thrust_thread = threading.Thread(target=thrust_control, daemon=True)
thrust_thread.start()

# Launch the viewer on the main thread.
with viewer.launch(model, data) as vis:
    while vis.is_running:
        vis.render()

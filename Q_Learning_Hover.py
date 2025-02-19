#------// Imports //------------------------------

import mujoco
from mujoco import viewer
import numpy as np
import time
import logging
import threading
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

def Reward_Function(current_flight_time, drone_x, drone_y, drone_z, desired_z):
    reward = (current_flight_time-10)*10  - 0.2*(abs(drone_x) + abs(drone_y)) - (abs(drone_z) - desired_z)*8 
    return reward


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
        raw_output = self.layer3(x)
        # Start at 6.5, allow network to output between 0-13
        return 6.5 + 6.5 * torch.tanh(raw_output)  # Maps to 0-13 centered at 6.5

#------// Setting Up Drone Environment   //------------------------------
# Load the model from the XML file
model = mujoco.MjModel.from_xml_path('skydio_x2/scene.xml')

# Create a data object
data = mujoco.MjData(model)

# Initialize control inputs to zero
data.ctrl[:] = 0.0

# Define thrust levels
max_thrust = 13
return_thrust = 3.4
increment = 0.5  # Define the increment step

# ---- Desired Altitude ----
desired_z = 10.0  # Set your desired altitude

def reset_simulation():
    """Resets the simulation to its initial state."""
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0  # Reset control inputs
    print("Simulation has been reset.")

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

#------// Action Mapping //------------------------------

# Define possible thrust adjustments corresponding to actions
ACTION_MAPPING = {
    0: -1.0,   # Decrease thrust by 1.0
    1: 0.0,    # Maintain current thrust
    2: 1.0,    # Increase thrust by 1.0
    3: max_thrust  # Set thrust to maximum value
}

#------// Selecting and Applying Actions //------------------------------

def main_loop():
    """Runs the combined training and thrust control loop."""
    global steps_done
    num_episodes = 600
    
    for i_episode in range(num_episodes):
        # Reset the Mujoco simulation
        reset_simulation()
        simulation_steps = 0  # Track simulation steps instead of wall clock time
        state = torch.tensor(get_state(), dtype=torch.float32, device=device).unsqueeze(0)

        episode_reward = 0
        action_counts = {0:0, 1:0, 2:0, 3:0}

        test = time.time()
        while True:
            # Select and perform an action
            action = select_action(state)
            action_counts[action.item()] += 1

            # Map the discrete action to thrust adjustment
            thrust_adjustment = ACTION_MAPPING[action.item()]

            # Get current thrust and apply adjustment
            current_thrust = data.ctrl[0]
            new_thrust = np.clip(current_thrust + thrust_adjustment, 0.0, max_thrust)

            data.ctrl[:] = [new_thrust] * model.nu
            mujoco.mj_step(model, data)
            simulation_steps += 1

            # Get new state
            next_state_np = get_state()
            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)

            # Get drone position
            drone_x = data.qpos[0]
            drone_y = data.qpos[1]
            drone_z = data.qpos[2]

            # Calculate reward using simulation time instead of wall clock time
            reward = Reward_Function((time.time() - test), drone_x, drone_y, drone_z, desired_z)
            #print(reward)
            episode_reward += reward
            current_sim_time = time.time() - test

            # Check if episode is done
            done = (current_sim_time > 2.0 and drone_z < 0.01) or \
                   (current_sim_time > 2.0 and drone_z > 15.0) or \
                   (current_sim_time > 2.0 and drone_x > 15.0) or \
                   (current_sim_time > 2.0 and drone_y > 15.0) or \
                   (current_sim_time > 2.0 and reward <= -200)
            
            if done:
                next_state = None
                reset_simulation()
                if reward <= -200:
                    print("Resetting due to low reward")
                logging.info(f"Action distribution in episode {i_episode + 1}: {action_counts}")

            # Store the transition in memory
            memory.push(state, action, next_state, torch.tensor([reward], device=device))
            state = next_state if not done else torch.tensor(get_state(), dtype=torch.float32, device=device).unsqueeze(0)

            # Optimize the model
            optimize_model()

            # Update the target network
            if steps_done % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                logging.info(f"Episode {i_episode + 1} finished with reward: {episode_reward:.2f}")
                logging.info(f"Time in simulation: {current_sim_time:.2f} seconds")
                break

    print('Training Complete')
    writer.close()

#------// Starting the Combined Loop and Launching Viewer   //------------------------------

if __name__ == "__main__":
    # Start the combined training and control loop
    training_thread = threading.Thread(target=main_loop, daemon=True)
    training_thread.start()

    # Launch the viewer on the main thread.
    with viewer.launch(model, data) as vis:
        while vis.is_running:
            mujoco.mj_step(model, data)  # Step the simulation
            vis.render()
            time.sleep(0.01)  # Control the simulation speed
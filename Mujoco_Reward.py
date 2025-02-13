import mujoco
from mujoco import viewer
import numpy as np
import time
import threading

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


def f(x):
    if x > 3.41:
        return 3.15  # Fix thrust at 2 after x = 3.41
    fx = -(x-2)**2 + 4
    return max(fx, 0)  # Simplified return statement

def thrust_control():
    current_time = 0.0
    start_time = time.time()  # Add start time for tracking flight duration
    while True:
        thrust_value = f(current_time)
        data.ctrl[:] = [thrust_value] * model.nu
        mujoco.mj_step(model, data)
        
        # Get drone position
        drone_x = data.qpos[0]
        drone_y = data.qpos[1]
        drone_z = data.qpos[2]
        
        # Calculate and print current time in air
        current_flight_time = time.time() - start_time
        if drone_z < 0.01:
            start_time = time.time()  # Add start time for tracking flight duration
        print(f"Time: {current_time:.2f}, Thrust: {thrust_value:.2f}, Position: x={drone_x:.2f}, y={drone_y:.2f}, z={drone_z:.2f}")
        print(f"Time in air: {current_flight_time:.2f}")
        print(f"Reward: {current_flight_time - (abs(drone_x) + abs(drone_y))*5}")

        current_time += 0.01
        if current_time > 10:
            current_time = 0
        time.sleep(0.01)

thrust_thread = threading.Thread(target=thrust_control, daemon=True)
thrust_thread.start()

# Launch the viewer on the main thread.
with viewer.launch(model, data) as vis:
    while vis.is_running:
        vis.render()

import mujoco
from mujoco import viewer
import numpy as np
import time
import threading

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path('../skydio_x2/scene.xml')

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
    # Increase thrust to max_thrust
    while True:
        thrust_value = f(current_time)  # Use the function f to determine thrust
        data.ctrl[:] = [thrust_value] * model.nu  # Set all control inputs to thrust_value
        mujoco.mj_step(model, data)
        print(f"Current time: {current_time:.2f}, Thrust value: {thrust_value:.2f}")  # Print time and thrust value
        current_time += 0.01
        if current_time > 10:
            current_time = 0
        time.sleep(0.01)  # Adjust as needed

# Start the thrust control in a separate thread.
thrust_thread = threading.Thread(target=thrust_control, daemon=True)
thrust_thread.start()

# Launch the viewer on the main thread.
with viewer.launch(model, data) as vis:
    while vis.is_running:
        vis.render()

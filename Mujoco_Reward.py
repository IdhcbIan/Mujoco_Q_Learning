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

def coef(x, a, b, c, O):
    if O:
        gx = a*np.sin(b*x) 
    else:
        gx = a*np.sin(b*x) + 0.3 * abs(np.sin(x/5))
    return abs(gx) + c

def f(x):
    fx = coef(x, 4, 1, 0, True) if x < np.pi/2 else coef(x, -0.96, 11, 2.6, False)
    return abs(fx)  # Simplified return statement

def reset_simulation():
    """
    Resets the simulation to its initial state.
    """
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0.0  # Reset control inputs
    print("Simulation has been reset.")

def thrust_control():
    start_time = time.time()  # Start time for tracking flight duration
    while True:
        current_flight_time = time.time() - start_time
        thrust_value = f(current_flight_time)  # Use elapsed time for thrust calculation
        data.ctrl[:] = [thrust_value] * model.nu
        mujoco.mj_step(model, data)
        
        # Get drone position
        drone_x = data.qpos[0]
        drone_y = data.qpos[1]
        drone_z = data.qpos[2]
        
        # Calculate and print current time in air
        # current_flight_time is already calculated
        
        # Only check for resets after the first second
        if current_flight_time > 8.0:
            if drone_z < 0.01:
                start_time = time.time()  # Reset start time for tracking flight duration
                reset_simulation()        # Reset the simulation when drone is on the ground
        
        # Calculate reward
        reward = current_flight_time - (abs(drone_x) + abs(drone_y)) * 5
        
        print(f"Time: {current_flight_time:.2f}, Thrust: {thrust_value:.2f}, Position: x={drone_x:.2f}, y={drone_y:.2f}, z={drone_z:.2f}")
        print(f"Time in air: {current_flight_time:.2f}")
        print(f"Reward: {reward}")
    
        # Only check for reward-based reset after the first second
        if current_flight_time > 2.0 and reward <= -20:
            start_time = time.time()
            reset_simulation()
            print("Resetting due to low reward")
    
        time.sleep(0.01)

thrust_thread = threading.Thread(target=thrust_control, daemon=True)
thrust_thread.start()

# Launch the viewer on the main thread.
with viewer.launch(model, data) as vis:
    while vis.is_running:
        vis.render()

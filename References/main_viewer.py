import mujoco
from mujoco import viewer
import numpy as np
import time
import threading
import socket
import json

# Configuration for IPC
HOST = 'localhost'
PORT = 50007  # Choose an appropriate port

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path('../skydio_x2/scene.xml')

# Create a data object
data = mujoco.MjData(model)

# Initialize control inputs to zero
data.ctrl[:] = 0.0

def handle_client_connection(conn, data):
    with conn:
        print('Connected by controller')
        while True:
            try:
                msg = conn.recv(1024)
                if not msg:
                    break
                # Assume the message is JSON encoded
                cmd = json.loads(msg.decode('utf-8'))
                if 'ctrl' in cmd:
                    # Update control inputs
                    control_values = cmd['ctrl']
                    if len(control_values) == model.nu:
                        data.ctrl[:] = control_values
                        print(f"Updated control inputs: {control_values}")
                    else:
                        print("Received control inputs with incorrect dimensions.")
            except ConnectionResetError:
                break
            except json.JSONDecodeError:
                print("Received invalid JSON.")
                continue

def start_ipc_server(model, data):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen()
    print(f"IPC Server listening on {HOST}:{PORT}")
    while True:
        conn, addr = server.accept()
        client_thread = threading.Thread(target=handle_client_connection, args=(conn, data), daemon=True)
        client_thread.start()

def main():
    # Start the IPC server in a separate thread.
    ipc_thread = threading.Thread(target=start_ipc_server, args=(model, data), daemon=True)
    ipc_thread.start()

    # Launch the viewer on the main thread.
    with viewer.launch(model, data) as vis:
        while vis.is_running:
            mujoco.mj_step(model, data)  # Step the simulation
            vis.render()

if __name__ == "__main__":
    main() 
import socket
import time
import json
import math

HOST = 'localhost'
PORT = 50007  # Must match the server's port

def f(x):
    fx = -(x-2)**2 + 4
    return max(fx, 0)  # Simplified return statement

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        current_time = 0.0
        while True:
            thrust_value = f(current_time)
            control_command = {
                'ctrl': [thrust_value]  # Adjust based on model.nu
            }
            msg = json.dumps(control_command).encode('utf-8')
            s.sendall(msg)
            print(f"Sent thrust value: {thrust_value:.2f}")
            current_time += 0.01
            if current_time > 10:
                current_time = 0
            time.sleep(0.01)

if __name__ == "__main__":
    main() 
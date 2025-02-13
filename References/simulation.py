import mujoco
import numpy as np
import time
import threading

class Simulation:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.data.ctrl[:] = 0.0
        self.max_thrust = 5
        self.return_thrust = 3
        self.increment = 0.1

    def f(self, x):
        fx = -(x - 2) ** 2 + 4
        return fx if fx > 0 else 0

    def thrust_control(self):
        current_time = 0.0
        while True:
            thrust_value = self.f(current_time)
            self.data.ctrl[:] = [thrust_value] * self.model.nu
            mujoco.mj_step(self.model, self.data)
            print(f"Thrust values set to: {thrust_value} at time {current_time}")
            current_time += 0.01
            if current_time > 10:
                current_time = 0
            time.sleep(0.01) 
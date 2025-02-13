import mujoco
from mujoco import viewer
import numpy as np
import time
import threading

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path('../skydio_x2/scene.xml')

# Create a data object
data = mujoco.MjData(model)




# Launch the viewer on the main thread.
with viewer.launch(model, data) as vis:
    while vis.is_running:
        vis.render()

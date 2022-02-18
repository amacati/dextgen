"""Mujoco installation test script."""

import mujoco_py as mj
from pathlib import Path

mj_path = mj.utils.discover_mujoco()
xml_path = Path(mj_path) / "model" / "humanoid.xml"
model = mj.load_model_from_path(str(xml_path))
sim = mj.MjSim(model)
try:
    mj_viewer = mj.MjViewerBasic(sim)
    display_available = True
except mj.cymj.GlfwError:
    display_available = False

for i in range(1000):
    sim.step()
    print(f"Sim step {i}")
    if display_available:
        mj_viewer.render()

print("Sim complete, Mujoco working as expected")

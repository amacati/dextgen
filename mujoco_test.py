import mujoco_py as mj 
from pathlib import Path


mj_path = mj.utils.discover_mujoco()
xml_path = Path(mj_path) / "model" / "humanoid.xml"
model = mj.load_model_from_path(str(xml_path))
sim = mj.MjSim(model)
mj_viewer = mj.MjViewerBasic(sim)

for i in range(1000):
    sim.step()
    print(f"Sim step {i}")
    mj_viewer.render()

from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Callable

from jax import jit
from jax.experimental import host_callback as hcb
import jax.numpy as jnp
import numpy as np

from optim.grippers.kinematics.tf import tf_matrix
from optim.grippers.kinematics.parallel_jaw import pj_full_kinematics, pj_kinematics, PJ_JOINT_LIMITS
from optim.geometry.rotations import mat2quat


class Gripper(ABC):

    POS_CNST = np.ones(3) * 2
    ORIENT_CNST = np.ones(4)

    def __init__(self, info: Dict):
        self.pos = np.array(info["gripper_info"]["pos"])
        self.orient_q = mat2quat(np.array(info["gripper_info"]["orient"]))
        self.grip_state = np.array(info["gripper_info"]["state"])
        self.state = np.concatenate((self.pos, self.orient_q, self.grip_state))

    @abstractmethod
    def create_kinematics(self, con_pt: Dict) -> Callable:
        ...

    @abstractproperty
    def joint_limits(self):
        ...

    def create_constraints(self, opt):
        # Unit quaternion constraint for orientation

        low_cnst = np.concatenate((-self.POS_CNST, -self.ORIENT_CNST, self.joint_limits["lower"]))
        high_cnst = np.concatenate((self.POS_CNST, self.ORIENT_CNST, self.joint_limits["upper"]))
        opt.set_lower_bounds(low_cnst)
        opt.set_upper_bounds(high_cnst)


class ParallelJaw(Gripper):

    KP = 30_000

    def __init__(self, info: Dict):
        super().__init__(info)

    @property
    def joint_limits(self):
        return PJ_JOINT_LIMITS

    def create_kinematics(self, link, con_pt: Dict) -> Callable:
        link_kinematics = pj_kinematics(link)
        finger_frame = link_kinematics(self.state)
        # Rotate dx into finger frame, add it to transformation chain
        dx = finger_frame[:3, :3].T @ (con_pt["pos"] - finger_frame[:3, 3])
        tf = tf_matrix(np.array([dx[0], dx[1], dx[2], 0, 0, 0]))

        @jit
        def kinematics(x):
            return (link_kinematics(x) @ tf)[:3, 3]

        return kinematics

    def create_full_kinematics(self, links, con_pts: Dict) -> Callable:
        links_idx = np.array([0 if link == "robot0:r_gripper_finger_link" else 1 for link in links])
        frames = pj_full_kinematics(self.state)
        f_pos = np.array([frames[idx][:3, 3] for idx in links_idx])
        f_rot = np.array([frames[idx][:3, :3] for idx in links_idx])
        dx = np.array([R.T @ (con_pt["pos"] - x) for con_pt, x, R in zip(con_pts, f_pos, f_rot)])
        tfs = np.array([tf_matrix(np.array([d[0], d[1], d[2], 0, 0, 0])) for d in dx])

        @jit
        def full_kinematics(x):
            frames = pj_full_kinematics(x)
            return jnp.array([(frame @ tf)[:3, 3] for frame, tf in zip(frames, tfs)])

        return full_kinematics

    def create_grasp_force(self, link):
        ...

    def create_grasp_forces(self, links, _):
        r_finger_kin = pj_kinematics("robot0:r_gripper_finger_link")
        l_finger_kin = pj_kinematics("robot0:l_gripper_finger_link")
        links_idx = np.array([1 if link == "robot0:r_gripper_finger_link" else 2 for link in links])

        def grasp_forces(x):
            forces = jnp.zeros((len(links), 6))
            # Skip computation via F = (J.T)^+ * tau because of gripper simplicity
            frame_r, frame_l = r_finger_kin(x), l_finger_kin(x)
            f = jnp.array([0, 0, self.KP * (x[-1] * -x[-2])])  # Simple P controller from MuJoCo
            # Rotate f from gripper frame to world frame
            fr, fl = frame_r[:3, :3] @ f, -frame_l[:3, :3] @ f  # fl works along negative z axis
            forces = forces.at[links_idx == 1, :3].set(fr)
            forces = forces.at[links_idx == 2, :3].set(fl)
            return forces

        return grasp_forces


class BarrettHand(Gripper):

    def __init__(self, info: Dict):
        super().__init__(info)


class ShadowHand(Gripper):

    def __init__(self, info: Dict):
        super().__init__(info)


def get_gripper(info: Dict) -> Gripper:
    gripper = info["gripper_info"]["type"]
    if gripper == "ParallelJaw":
        return ParallelJaw(info)
    elif gripper == "BarrettHand":
        return BarrettHand(info)
    elif gripper == "ShadowHand":
        return ShadowHand(info)
    raise RuntimeError(f"Gripper {gripper} not supported")

from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, Callable

from jax import jit
import jax.numpy as jnp
import numpy as np

from optim.grippers.kinematics.tf import tf_matrix
from optim.grippers.kinematics.parallel_jaw import pj_full_kinematics, pj_kinematics
from optim.grippers.kinematics.parallel_jaw import PJ_JOINT_LIMITS
from optim.rotations import mat2quat


class Gripper(ABC):

    POS_CNST = np.ones(3) * 2
    ORIENT_CNST = np.ones(4)

    def __init__(self, info: Dict):
        self.pos = np.array(info["gripper_info"]["pos"])
        self.orient_q = mat2quat(np.array(info["gripper_info"]["orient"]))
        self.grip_state = np.array(info["gripper_info"]["state"])
        self.state = np.concatenate((self.pos, self.orient_q, self.grip_state))
        self.con_links = np.array([i["geom1"] for i in info["contact_info"]])

    @abstractmethod
    def create_kinematics(self, con_pt: Dict) -> Callable:
        ...

    @abstractproperty
    def joint_limits(self):
        ...

    def create_constraints(self, opt, localopt):
        # Unit quaternion constraint for orientation
        low, high = np.tile(self.joint_limits["lower"], 2), np.tile(self.joint_limits["upper"], 2)
        low_cnst = np.concatenate((-self.POS_CNST, -self.ORIENT_CNST, low))
        high_cnst = np.concatenate((self.POS_CNST, self.ORIENT_CNST, high))
        opt.set_lower_bounds(low_cnst)
        opt.set_upper_bounds(high_cnst)
        localopt.set_lower_bounds(low_cnst)
        localopt.set_upper_bounds(high_cnst)


class ParallelJaw(Gripper):

    KP = 30_000
    LINKS = ("robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link")

    def __init__(self, info: Dict):
        super().__init__(info)

    @property
    def joint_limits(self):
        return PJ_JOINT_LIMITS

    def create_kinematics(self, link, con_pt: Dict) -> Callable:
        link_kinematics = pj_kinematics(link)
        # finger_frame = link_kinematics(self.state)
        # Rotate dx into finger frame, add it to transformation chain
        # dx = finger_frame[:3, :3].T @ (con_pt["pos"] - finger_frame[:3, 3])
        # tf = tf_matrix(np.array([dx[0], dx[1], dx[2], 0, 0, 0]))

        @jit
        def kinematics(x):
            return link_kinematics(x)[:3, 3]

        return kinematics

    def create_full_kinematics(self, links, _) -> Callable:
        f_idx = np.array([-2 if link == self.LINKS[0] else -1 for link in links])
        # frames = pj_full_kinematics(self.state)[-2:]
        # f_pos = np.array([frames[idx][:3, 3] for idx in links_idx])
        # f_rot = np.array([frames[idx][:3, :3] for idx in links_idx])
        # dx = np.array([R.T @ (con_pt["pos"] - x) for con_pt, x, R in zip(con_pts, f_pos, f_rot)])
        # tfs = np.array([tf_matrix(np.array([d[0], d[1], d[2], 0, 0, 0])) for d in dx])

        @jit
        def full_kinematics(x):
            frames = pj_full_kinematics(x)[-2:]
            return jnp.array([frames[i][:3, 3] for i in f_idx])
            # return jnp.array([(frame @ tf)[:3, 3] for frame, tf in zip(frames, tfs)])

        return full_kinematics

    def create_full_frames(self, links, _) -> Callable:
        f_idx = np.array([-2 if link == self.LINKS[0] else -1 for link in links])

        @jit
        def full_kinematics(x):
            frames = pj_full_kinematics(x)[-2:]
            return jnp.array([frames[i] for i in f_idx])

        return full_kinematics

    def create_grasp_force(self, link):
        assert link in self.LINKS
        kinematics = pj_kinematics(link)
        link_axis = 1 if link == self.LINKS[0] else -1  # left gripper moves along -z axis

        @jit
        def grasp_force(x):
            frame = kinematics(x)
            f = frame[:3, :3] @ jnp.array([0, link_axis * self.KP * (x[-1] - x[-2]), 0])
            return jnp.array([*f, 0, 0, 0])  # Wrench with zero torque elements

        return grasp_force

    def create_grasp_forces(self, links, _):
        r_finger_kin = pj_kinematics("robot0:r_gripper_finger_link")
        l_finger_kin = pj_kinematics("robot0:l_gripper_finger_link")
        links_idx = np.array([1 if link == "robot0:r_gripper_finger_link" else 2 for link in links])

        @jit
        def grasp_forces(x):
            forces = jnp.zeros((len(links), 6))
            # Skip computation via F = (J.T)^+ * tau because of gripper simplicity
            frame_r, frame_l = r_finger_kin(x), l_finger_kin(x)
            f = jnp.array([0, self.KP * (x[-1] - x[-2]), 0])  # Simple P controller from MuJoCo
            # Rotate f from gripper frame to world frame
            fr, fl = frame_r[:3, :3] @ f, -frame_l[:3, :3] @ f  # fl works along negative y axis
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

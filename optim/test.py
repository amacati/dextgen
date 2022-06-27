import numpy as np
from optim.kinematics.sh_kinematics import sh_kinematics
from optim.kinematics.pj_kinematics import pj_kinematics
from optim.kinematics.bh_kinematics import bh_kinematics
from optim.kinematics.visualize import visualize_frames


def main():
    x = np.zeros(10)
    x[6] = 1
    frames = pj_kinematics(x)
    visualize_frames(frames, "pj")


if __name__ == "__main__":
    main()

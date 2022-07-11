from functools import singledispatch

from optim.grippers import ParallelJaw, BarrettHand, ShadowHand
from optim.grippers.kinematics.parallel_jaw import pj_full_kinematics


@singledispatch
def visualize_gripper(gripper, fig, color="#777777"):
    ...


@visualize_gripper.register
def _(gripper: ParallelJaw, fig, color="#777777"):
    frames = pj_full_kinematics(gripper.state)
    frame_pairs = {1: 0, 2: 0}
    return _visualize_frames(fig, frames, frame_pairs, color)


@visualize_gripper.register
def _(gripper: BarrettHand, fig, color="#777777"):
    raise NotImplementedError


@visualize_gripper.register
def _(gripper: ShadowHand, fig, color="#777777"):
    raise NotImplementedError


def _visualize_frames(fig, frames, _, color="#777777"):
    ax = fig.axes[0]
    for frame in frames:
        pos = frame[0:3, 3]
        scl = .01
        ex, ey, ez = frame[0:3, 0] * scl + pos, frame[0:3, 1] * scl + pos, frame[0:3, 2] * scl + pos
        ax.scatter(*pos, color="0")
        for ei, col in zip([ex, ey, ez], ["r", "g", "b"]):
            ax.plot([pos[0], ei[0]], [pos[1], ei[1]], zs=[pos[2], ei[2]], color=col)
    for i in range(1, 3):
        pos = frames[i][0:3, 3]
        glow, ghigh = pos + frame[0:3, 0] * 0.04, pos - frame[0:3, 0] * 0.04
        ax.plot([glow[0], ghigh[0]], [glow[1], ghigh[1]], [glow[2], ghigh[2]], color=color)
    gright = frames[1][:3, 3] + frames[1][:3, 0] * 0.04
    gleft = frames[2][:3, 3] + frames[1][:3, 0] * 0.04
    gmiddle = (gleft + gright) / 2
    gtop = frames[0][:3, 3]
    ax.plot([gright[0], gleft[0]], [gright[1], gleft[1]], [gright[2], gleft[2]], color=color)
    ax.plot([gmiddle[0], gtop[0]], [gmiddle[1], gtop[1]], [gmiddle[2], gtop[2]], color=color)
    return fig

"""Gripper visualization module."""
from __future__ import annotations
from functools import singledispatch
from typing import Optional, List

import matplotlib.pyplot as plt

from optim.grippers.kinematics.parallel_jaw import kin_pj_full
from optim.grippers.kinematics.barrett_hand import kin_bh_full
from optim.utils.utils import import_guard

if import_guard():
    from matplotlib.figure import Figure  # noqa: TC002, is guarded
    from optim.grippers import ParallelJaw, BarrettHand, ShadowHand  # noqa: TC001, is guarded


@singledispatch
def visualize_gripper(gripper, fig: Figure, color: str = "#777777") -> Figure:
    """Dispatch function for plotting a gripper.

    Args:
        gripper: Gripper.
        fig: Optional figure to plot into.
        color: Optional matplotlib compatible color string.

    Returns:
        The figure.

    Raises:
        RuntimeError: Gripper type is not supported.
    """
    raise RuntimeError(f"Gripper of type {type(gripper)} not supported")


@visualize_gripper.register
def _(gripper: ParallelJaw, fig: Optional[Figure] = None, color: str = "#777777") -> Figure:
    """Plot a ParallelJaw gripper.

    Args:
        gripper: Gripper.
        fig: Optional figure to plot into.
        color: Optional matplotlib compatible color string.

    Returns:
        The figure.
    """
    if fig is None:
        fig = _get_fig()
    frames = kin_pj_full(gripper.state)
    ax = fig.axes[0]
    for i, frame in enumerate(frames):
        pos = frame[:3, 3]
        scl = .01
        ex, ey, ez = frame[:3, 0] * scl + pos, frame[:3, 1] * scl + pos, frame[:3, 2] * scl + pos
        if i > 0:
            ax.scatter(*pos, color=("r", "r", "g", "g")[i - 1])
        for ei, col in zip([ex, ey, ez], ["r", "g", "b"]):
            ax.plot([pos[0], ei[0]], [pos[1], ei[1]], zs=[pos[2], ei[2]], color=col)
    for i in range(1, 5):
        dx1, dx2 = (0.0385, 0.0385) if i % 2 else (0.05775, 0.01925)
        pos = frames[i][0:3, 3]
        glow, ghigh = pos + frame[0:3, 0] * dx1, pos - frame[0:3, 0] * dx2
        ax.plot([glow[0], ghigh[0]], [glow[1], ghigh[1]], [glow[2], ghigh[2]], color=color)
    gright = frames[1][:3, 3] + frames[1][:3, 0] * 0.0385
    gleft = frames[3][:3, 3] + frames[3][:3, 0] * 0.0385
    gmiddle = (gleft + gright) / 2
    gtop = frames[0][:3, 3]
    ax.plot([gright[0], gleft[0]], [gright[1], gleft[1]], [gright[2], gleft[2]], color=color)
    ax.plot([gmiddle[0], gtop[0]], [gmiddle[1], gtop[1]], [gmiddle[2], gtop[2]], color=color)
    return fig


@visualize_gripper.register
def _(gripper: BarrettHand, fig: Optional[Figure] = None, color: str = "#777777") -> Figure:
    """Plot a BarrettHand.

    Args:
        gripper: Gripper.
        fig: Optional figure to plot into.
        color: Optional matplotlib compatible color string.

    Returns:
        The figure.
    """
    if fig is None:
        fig = _get_fig()
    frames = kin_bh_full(gripper.state)
    ax = fig.axes[0]
    for i, frame in enumerate(frames):
        pos = frame[:3, 3]
        scl = .01
        ex, ey, ez = frame[:3, 0] * scl + pos, frame[:3, 1] * scl + pos, frame[:3, 2] * scl + pos
        for ei, col in zip([ex, ey, ez], ["r", "g", "b"]):
            ax.plot([pos[0], ei[0]], [pos[1], ei[1]], zs=[pos[2], ei[2]], color=col)
    ps = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11))
    for (i, j) in ps:
        pos1, pos2 = frames[i][0:3, 3], frames[j][0:3, 3]
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], color=color)
    return fig


@visualize_gripper.register
def _(gripper: ShadowHand, fig: Optional[Figure] = None, color: str = "#777777"):
    """Plot a ShadowHand.

    Args:
        gripper: Gripper.
        fig: Optional figure to plot into.
        color: Optional matplotlib compatible color string.

    Returns:
        The figure.

    Raises:
        NotImplementedError: Function is currently not implemented.
    """
    raise NotImplementedError


def _get_fig(limits: Optional[List] = None) -> Figure:
    """Create a figure to plot into.

    Args:
        limits: Optional plot axis limits.

    Returns:
        The figure.
    """
    fig = plt.figure()
    fig.suptitle("Contact point optimization")
    ax = []
    ax.append(fig.add_subplot(111, projection="3d"))
    ax[0].set_box_aspect(aspect=(1, 1, 1))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")
    if limits is not None:
        ax[0].set_xlim(limits[0])
        ax[0].set_ylim(limits[1])
        ax[0].set_zlim(limits[2])
    return fig

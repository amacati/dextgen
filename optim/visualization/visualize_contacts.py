"""Contact visualization module."""
from __future__ import annotations
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

from optim.utils.utils import import_guard

if import_guard():
    from matplotlib.figure import Figure  # noqa: TC002, is guarded


def visualize_contacts(con_pts: List[Dict], fig: Optional[Figure] = None) -> Figure:
    """Visualize the contact points.

    Only plots into a figure. `plt.show` has to be invoked manually.

    Args:
        con_pts: Contact point list.
        fig: Optional figure to plot the contact points into. Allows to plot multiple objects.

    Returns:
        The figure.
    """
    if fig is None:
        fig = _get_fig()
    ax = fig.axes[0]
    for con_pt in con_pts:
        ax.scatter(*con_pt["pos"], color="r", s=2)
    return fig


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

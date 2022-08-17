"""Geometry visualization module."""
from __future__ import annotations
from functools import singledispatch

import numpy as np
import matplotlib.pyplot as plt

from optim.utils.utils import import_guard

if import_guard():
    from matplotlib.figure import Figure  # noqa: TC002, is guarded
    from optim.geometry import Cube, Geometry  # noqa: TC001, is guarded


@singledispatch
def visualize_geometry(geom: Geometry):
    """Visualize dispatch function for geometries.

    Args:
        geom: Geometry.

    Raises:
        TypeError: Default dispatcher means an unsupported geometry has been used.
    """
    raise TypeError(f"Geometry type {type(geom)} not supported")


@visualize_geometry.register
def _(geom: Cube) -> Figure:
    """Visualize the cube geometry.

    Args:
        geom: Cube geometry.

    Returns:
        The figure.
    """
    fig = plt.figure()
    fig.suptitle("Contact point optimization")
    ax = []
    ax.append(fig.add_subplot(111, projection="3d"))
    dst = np.linalg.norm(geom.plane_offsets[0, 1] - geom.plane_offsets[0, 2])
    ax[0].set_title("cube")
    ax[0].set_box_aspect(aspect=(1, 1, 1))
    ax[0].set_xlim(geom.com[0] + dst * np.array([-2, 2]))
    ax[0].set_ylim(geom.com[1] + dst * np.array([-2, 2]))
    ax[0].set_zlim(geom.com[2] + dst * np.array([-2, 2]))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")

    for k, offsets in enumerate(geom.plane_offsets):
        for i in range(1, len(offsets)):
            j = 3 if i < 3 else 1
            p0 = offsets[i] + (offsets[j] - offsets[0])
            p1 = offsets[i] + (offsets[j + 1] - offsets[0])
            ax[0].plot3D(*zip(p0, p1), color="k")
        ax[0].scatter(*offsets[0], color=("r", "g", "b", "c", "m", "y")[k])
    # Plot table surface
    xsupport = np.linspace(*(geom.com[0] + dst * np.array([-1, 1])), 2)
    ysupport = np.linspace(*(geom.com[1] + dst * np.array([-1, 1])), 2)
    x, y = np.meshgrid(xsupport, ysupport)
    z = np.ones_like(x) * 0.4
    ax[0].plot_surface(x, y, z, alpha=0.4)
    return fig

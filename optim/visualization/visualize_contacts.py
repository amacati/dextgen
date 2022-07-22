import matplotlib.pyplot as plt


def visualize_contacts(con_pts, fig=None):
    if fig is None:
        fig = _get_fig()
    ax = fig.axes[0]
    for con_pt in con_pts:
        ax.scatter(*con_pt["pos"], color="r", s=2)
    return fig


def _get_fig(limits=None):
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

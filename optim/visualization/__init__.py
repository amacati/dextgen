import matplotlib.pyplot as plt

from optim.visualization.visualize_gripper import visualize_gripper
from optim.visualization.visualize_object import visualize_object
from optim.visualization.visualize_contacts import visualize_contacts


def visualize_grasp(obj, gripper, x):
    fig = visualize_object(obj)
    gripper.state = x
    fig = visualize_gripper(gripper, fig)
    fig = visualize_contacts(fig, obj, gripper)
    return fig

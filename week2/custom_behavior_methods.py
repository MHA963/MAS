import numpy as np
from irsim.lib import register_behavior
from irsim.util.util import WrapToPi

@register_behavior("diff", "circle_follow")
def beh_diff_circle_follow(ego_object, external_objects=None, **kwargs):
    """
    Custom behavior: make the robot follow a circle around CENTER with RADIUS
    """
    if external_objects is None:
        external_objects = []

    state = ego_object.state  # [x, y, theta]
    max_vel = ego_object.get_vel_range()[1]  # max linear/angular velocity

    # Circle parameters
    CENTER = np.array(kwargs.get("center", [5.0, 5.0]))
    RADIUS = kwargs.get("radius", 2.0)

    return CircleFollow(state, max_vel, CENTER, RADIUS)


def CircleFollow(state, max_vel, center, radius):
    """
    Calculate linear and angular velocity for circular motion.
    """
    x, y, theta = state.flatten()
    vec_to_center = center - np.array([x, y])
    distance = np.linalg.norm(vec_to_center)

    # Tangential direction along circle (CCW)
    angle_to_center = np.arctan2(vec_to_center[1], vec_to_center[0])
    tangential_angle = angle_to_center + np.pi/2

    # Angle difference
    angle_diff = WrapToPi(tangential_angle - theta)

    # Linear speed: constant fraction of max velocity, minimum 0.1
    linear_speed = max(0.1, max_vel[0, 0]*0.5)

    # Angular speed proportional to angle difference (smooth with tanh)
    angular_speed = max_vel[1, 0] * np.tanh(2 * angle_diff)

    return np.array([[linear_speed], [angular_speed]])

"""Render with PyBullet."""

try:
    import pybullet
except ImportError as e:
    err = (
        "You need the 'render' extra option to use this module.\n"
        "For this, you can install the 'happypose[render]' package."
    )
    raise ImportError(err) from e

from .renderer import Renderer


class PyBulletRenderer(Renderer):
    """PyBullet renderer."""

    def __init__(self, w=10, h=10, fov=60, near=0.01, far=100):
        """Initialize the required parameters for rendering with pybullet."""
        self.w = w
        self.h = h
        self.fov = fov
        self.near = near
        self.far = far

    def render(self, scene):
        """Render the scene and return an image."""
        projection_matrix = pybullet.computeProjectionMatrixFOV(
            self.fov,
            self.aspect,
            self.near,
            self.far,
        )
        _, _, img, _ = pybullet.getCameraImage(
            self.w,
            self.h,
            scene,
            projection_matrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )
        return img

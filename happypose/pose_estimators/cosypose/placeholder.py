"""Placeholder file to serve as example."""

from typing import Optional

from happypose.toolbox.renderer import Renderer


def cosy_placeholder(renderer: Optional[Renderer] = None):
    """Mock a call to cosypose with an optional rendering.

    This make no sense, other than providing content to the layout of the project.

    >>> cosy_placeholder()
    42
    >>> from happypose.toolbox.renderer.pybullet import PyBulletRenderer
    >>> render = PyBulletRenderer()
    >>> cosy_placeholder(render)
    100
    """
    return 42 if renderer is None else renderer.far

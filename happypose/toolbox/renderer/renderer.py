"""Base."""

from abc import ABC, abstractmethod


class Renderer(ABC):
    """Abstract base class for renderers."""

    @abstractmethod
    def render(self, scene):
        """Render the scene and return an image."""
        pass

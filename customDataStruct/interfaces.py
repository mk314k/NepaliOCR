"""
Abstract base interface for axis-aligned 2D bounding rectangles.
"""


class Rectangle:
    """Interface for 2D rectangular regions supporting overlap and union updates."""

    def isOverlapping(self, rect: object, padding: int = 0) -> bool:
        """Return True if `self` overlaps `rect` within `padding` pixels."""
        raise NotImplementedError

    def update(
        self, rect: object, padding: int = 0, paddedUpdate: bool = True
    ) -> None:
        """Expand `self` in-place to the bounding box union with `rect`."""
        raise NotImplementedError

    def overlapUpdate(self, rect: object, padding: int = 0) -> bool:
        """Merge `rect` into `self` if they overlap; return True when merged."""
        raise NotImplementedError

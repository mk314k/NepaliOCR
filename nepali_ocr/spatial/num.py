"""
Mutable integer wrapper used for shared index references in `RectSet` grid blocks.
"""


class Num:
    """Boxed integer value so multiple spatial-grid cells can reference the same slot."""

    def __init__(self, num: int = 0) -> None:
        self.num = int(num)

    def set(self, num: int) -> None:
        self.num = int(num)

    def __call__(self) -> int:
        return self.num

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Num):
            return self.num == other.num
        if isinstance(other, int):
            return self.num == other
        return False

    def __repr__(self) -> str:
        return f"Num({self.num})"

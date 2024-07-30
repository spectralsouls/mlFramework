from typing import Tuple
from dataclasses import dataclass

@dataclass
class View:
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    offset: int
    contiguous: bool

    @staticmethod
    def create(shape:Tuple[int, ...], stride: Tuple[int, ...], offset:int=0, contiguous:bool=True):
        return View(shape, stride, offset, contiguous)

@dataclass
class ShapeTracker:
    views: View
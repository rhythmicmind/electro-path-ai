"""
Core Geometry & Spatial Representation Module for Electrical Layouts.
Models 2D/3D building structures, rooms, wall segments, openings, and materials.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any


class WallMaterial(Enum):
    DRYWALL = "drywall"                      # Easy to fish wires through hollow cavities
    LIGHT_PARTITION = "light_partition"      # Stud partition
    MASONRY_BRICK = "masonry_brick"          # Requires surface chasing / grooving
    REINFORCED_CONCRETE = "reinforced_concrete"  # Extremely difficult to groove; prefer ceiling drops


# Cost multipliers for routing conduits through different substrates
SUBSTRATE_ROUTING_COST = {
    WallMaterial.DRYWALL: 1.0,
    WallMaterial.LIGHT_PARTITION: 1.2,
    WallMaterial.MASONRY_BRICK: 2.2,
    WallMaterial.REINFORCED_CONCRETE: 4.5,
}


class OpeningType(Enum):
    DOOR = "door"
    WINDOW = "window"
    STRUCTURAL_COLUMN = "structural_column"
    PASSAGEWAY = "passageway"


class RoomType(Enum):
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    CORRIDOR = "corridor"
    UTILITY = "utility"
    BALCONY = "balcony"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def manhattan_distance_to(self, other: Point2D) -> float:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __add__(self, other: Point2D) -> Point2D:
        return Point2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point2D) -> Point2D:
        return Point2D(self.x - other.x, self.y - other.y)

    def __lt__(self, other: Point2D) -> bool:
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    def round(self, decimals: int = 3) -> Point2D:
        return Point2D(round(self.x, decimals), round(self.y, decimals))


@dataclass
class Opening:
    opening_type: OpeningType
    offset: float        # Distance from wall start point along the segment
    width: float         # Width of the opening (meters)
    height: float = 2.1  # Height of opening (meters)
    latch_side: str = "right"  # "left" or "right" for doors


@dataclass
class WallSegment:
    id: str
    start: Point2D
    end: Point2D
    thickness: float = 0.15                  # meters
    height: float = 2.8                      # ceiling height (meters)
    material: WallMaterial = WallMaterial.MASONRY_BRICK
    is_exterior: bool = False
    openings: List[Opening] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def direction(self) -> Point2D:
        L = self.length
        if L < 1e-6:
            return Point2D(0.0, 0.0)
        return Point2D((self.end.x - self.start.x) / L, (self.end.y - self.start.y) / L)

    @property
    def normal(self) -> Point2D:
        # Perpendicular normal unit vector
        d = self.direction
        return Point2D(-d.y, d.x)

    def point_at(self, offset: float) -> Point2D:
        """Returns Point2D at given linear distance from wall start."""
        d = self.direction
        return Point2D(self.start.x + d.x * offset, self.start.y + d.y * offset).round(3)

    def get_unobstructed_intervals(self, clearance: float = 0.1) -> List[Tuple[float, float]]:
        """
        Calculates wall intervals where receptacles/switches can be safely placed,
        excluding door jambs, windows, and structural obstacles.
        """
        blocked_spans: List[Tuple[float, float]] = []
        for op in self.openings:
            start_span = max(0.0, op.offset - clearance)
            end_span = min(self.length, op.offset + op.width + clearance)
            blocked_spans.append((start_span, end_span))

        # Sort and merge overlapping blocked spans
        blocked_spans.sort(key=lambda s: s[0])
        merged_blocks: List[Tuple[float, float]] = []
        for span in blocked_spans:
            if not merged_blocks or span[0] > merged_blocks[-1][1]:
                merged_blocks.append(span)
            else:
                merged_blocks[-1] = (merged_blocks[-1][0], max(merged_blocks[-1][1], span[1]))

        # Invert to find free intervals
        free_intervals: List[Tuple[float, float]] = []
        curr = 0.0
        for b_start, b_end in merged_blocks:
            if b_start > curr:
                free_intervals.append((curr, b_start))
            curr = max(curr, b_end)
        if curr < self.length:
            free_intervals.append((curr, self.length))

        return free_intervals


@dataclass
class Room:
    id: str
    name: str
    room_type: RoomType
    walls: List[WallSegment] = field(default_factory=list)
    ceiling_height: float = 2.8

    @property
    def centroid(self) -> Point2D:
        if not self.walls:
            return Point2D(0.0, 0.0)
        xs = [w.start.x for w in self.walls] + [w.end.x for w in self.walls]
        ys = [w.start.y for w in self.walls] + [w.end.y for w in self.walls]
        return Point2D(sum(xs) / len(xs), sum(ys) / len(ys)).round(3)

    @property
    def perimeter(self) -> float:
        return sum(w.length for w in self.walls)


@dataclass
class FloorPlan:
    name: str
    rooms: Dict[str, Room] = field(default_factory=dict)
    all_walls: List[WallSegment] = field(default_factory=list)
    panel_location: Optional[Point2D] = None  # Main Distribution Board (MDB)

    def add_room(self, room: Room) -> None:
        self.rooms[room.id] = room
        for w in room.walls:
            if w not in self.all_walls:
                self.all_walls.append(w)

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        if not self.all_walls:
            return (0.0, 0.0, 10.0, 10.0)
        xs = [w.start.x for w in self.all_walls] + [w.end.x for w in self.all_walls]
        ys = [w.start.y for w in self.all_walls] + [w.end.y for w in self.all_walls]
        return (min(xs), min(ys), max(xs), max(ys))

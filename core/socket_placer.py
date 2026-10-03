"""
Socket and Fixture Placement Engine.
Applies statutory electrical codes (NEC/IEC) and spatial reasoning to place
outlets, switches, luminaires, and distribution panels along walls and ceilings.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from core.geometry import (
    Point2D, WallSegment, Room, FloorPlan, RoomType, OpeningType, WallMaterial
)
from core.standards import (
    FixtureType, FIXTURE_SPECS, FixtureSpec, ElectricalStandards
)


@dataclass
class ElectricalNode:
    id: str
    fixture_type: FixtureType
    location: Point2D
    room_id: str
    wall_id: Optional[str] = None
    wall_offset: Optional[float] = None
    circuit_id: Optional[str] = None
    description: str = ""

    @property
    def spec(self) -> FixtureSpec:
        return FIXTURE_SPECS[self.fixture_type]

    @property
    def load_watts(self) -> float:
        return self.spec.typical_load_watts


class SocketPlacementEngine:
    """
    Automated code-compliant placement engine.
    Computes unobstructed wall intervals and generates optimal fixture nodes.
    """

    def __init__(self, floor_plan: FloorPlan):
        self.floor_plan = floor_plan
        self.placed_nodes: List[ElectricalNode] = []
        self._node_counter: int = 0

    def _next_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"{prefix}_{self._node_counter:03d}"

    def place_all(self) -> List[ElectricalNode]:
        """Runs full placement pipeline for the floor plan."""
        self.placed_nodes.clear()
        self._node_counter = 0

        # 1. Place Main Distribution Panel if not already set
        self._place_main_panel()

        # 2. Place fixtures room by room
        for room_id, room in self.floor_plan.rooms.items():
            self._place_room_fixtures(room)

        return self.placed_nodes

    def _place_main_panel(self) -> None:
        """Determines optimal placement for the Main Distribution Board (MDB)."""
        if self.floor_plan.panel_location:
            panel_node = ElectricalNode(
                id="MDB_PANEL",
                fixture_type=FixtureType.MAIN_PANEL,
                location=self.floor_plan.panel_location,
                room_id="SYSTEM",
                description="Main Distribution Board (200A 120/240V Service Panel)",
            )
            self.placed_nodes.append(panel_node)
            return

        # Heuristic: place in Utility room or Corridor near an exterior wall
        candidate_wall = None
        target_room_id = "SYSTEM"

        for r_id, room in self.floor_plan.rooms.items():
            if room.room_type in [RoomType.UTILITY, RoomType.CORRIDOR]:
                for w in room.walls:
                    if w.is_exterior or not candidate_wall:
                        candidate_wall = w
                        target_room_id = r_id
                        break

        if not candidate_wall and self.floor_plan.all_walls:
            candidate_wall = self.floor_plan.all_walls[0]
            target_room_id = "SYSTEM"

        if candidate_wall:
            loc = candidate_wall.point_at(min(0.8, candidate_wall.length / 2))
            panel_node = ElectricalNode(
                id="MDB_PANEL",
                fixture_type=FixtureType.MAIN_PANEL,
                location=loc,
                room_id=target_room_id,
                wall_id=candidate_wall.id,
                wall_offset=0.8,
                description="Main Distribution Board (200A 120/240V Service Panel)",
            )
            self.floor_plan.panel_location = loc
            self.placed_nodes.append(panel_node)

    def _place_room_fixtures(self, room: Room) -> None:
        """Places switches, luminaires, and outlets for a single room."""
        # 1. Ceiling Luminaire at Room Centroid
        centroid = room.centroid
        self.placed_nodes.append(ElectricalNode(
            id=self._next_id("LIGHT"),
            fixture_type=FixtureType.CEILING_LIGHT,
            location=centroid,
            room_id=room.id,
            description=f"Ceiling luminaire for {room.name}",
        ))

        # 2. Door Switches & Outlets along walls
        for wall in room.walls:
            # Check for door openings to place entry light switch
            for op in wall.openings:
                if op.opening_type == OpeningType.DOOR:
                    self._place_door_switch(room, wall, op)

            # Place wall receptacles based on room type
            self._place_wall_receptacles(room, wall)

        # 3. Room-Specific Special Receptacles
        if room.room_type == RoomType.KITCHEN:
            self._place_kitchen_dedicated_appliances(room)

    def _place_door_switch(self, room: Room, wall: WallSegment, door: Opening) -> None:
        """Places single-pole switch on the latch side of the door."""
        offset = door.offset + door.width + ElectricalStandards.DOOR_SWITCH_JAMB_OFFSET_M
        if offset > wall.length:
            offset = max(0.1, door.offset - ElectricalStandards.DOOR_SWITCH_JAMB_OFFSET_M)

        loc = wall.point_at(offset)
        self.placed_nodes.append(ElectricalNode(
            id=self._next_id("SW"),
            fixture_type=FixtureType.SWITCH_SINGLE_POLE,
            location=loc,
            room_id=room.id,
            wall_id=wall.id,
            wall_offset=offset,
            description=f"Entry switch for {room.name}",
        ))

    def _place_wall_receptacles(self, room: Room, wall: WallSegment) -> None:
        """
        Samples receptacles along unobstructed spans to meet NEC 210.52 regulations.
        """
        intervals = wall.get_unobstructed_intervals(clearance=0.15)
        is_kitchen = (room.room_type == RoomType.KITCHEN)
        is_bathroom = (room.room_type == RoomType.BATHROOM)

        if is_kitchen or is_bathroom:
            max_spacing = ElectricalStandards.KITCHEN_COUNTER_MAX_SPACING_M
            max_corner = ElectricalStandards.KITCHEN_COUNTER_EDGE_DIST_M
            fixture_t = FixtureType.GFCI_20A
            prefix = "GFCI"
        else:
            max_spacing = ElectricalStandards.GENERAL_MAX_SPACING_M
            max_corner = ElectricalStandards.GENERAL_CORNER_MAX_DIST_M
            fixture_t = FixtureType.DUPLEX_15A
            prefix = "REC"

        for start_t, end_t in intervals:
            span_len = end_t - start_t
            if span_len < ElectricalStandards.GENERAL_WALL_MIN_LENGTH_M:
                continue

            # Compute number of outlets needed to satisfy:
            # no point is more than max_corner from an outlet
            if span_len <= 2 * max_corner:
                # 1 outlet at center of span suffices
                offsets = [start_t + span_len / 2.0]
            else:
                num_outlets = math.ceil((span_len - 2 * max_corner) / max_spacing) + 1
                step = span_len / (num_outlets + 1)
                offsets = [start_t + step * (i + 1) for i in range(num_outlets)]

            for off in offsets:
                loc = wall.point_at(off)
                self.placed_nodes.append(ElectricalNode(
                    id=self._next_id(prefix),
                    fixture_type=fixture_t,
                    location=loc,
                    room_id=room.id,
                    wall_id=wall.id,
                    wall_offset=off,
                    description=f"{fixture_t.value.upper()} on wall {wall.id} in {room.name}",
                ))

    def _place_kitchen_dedicated_appliances(self, room: Room) -> None:
        """Places dedicated high-amperage lines for refrigerator, microwave, and electric range."""
        if not room.walls:
            return

        w = room.walls[0]
        # Refrigerator dedicated 20A
        refrig_loc = w.point_at(min(0.5, w.length * 0.2))
        self.placed_nodes.append(ElectricalNode(
            id=self._next_id("DEDICATED_FRIDGE"),
            fixture_type=FixtureType.DEDICATED_20A,
            location=refrig_loc,
            room_id=room.id,
            wall_id=w.id,
            description="Dedicated 20A circuit for Refrigerator",
        ))

        # Electric Range / Oven (240V 50A)
        range_loc = w.point_at(min(1.8, w.length * 0.5))
        self.placed_nodes.append(ElectricalNode(
            id=self._next_id("HEAVY_RANGE"),
            fixture_type=FixtureType.HEAVY_DUTY_240V,
            location=range_loc,
            room_id=room.id,
            wall_id=w.id,
            description="Dedicated 240V 50A circuit for Electric Range / Cooktop",
        ))

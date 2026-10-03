"""
Vision & Spatial Reconstruction Interface.
Processes floor plan blueprints and cement wall site photos to extract
metric wall planes, structural keep-out zones, and room topologies.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

from core.geometry import (
    Point2D, WallSegment, Opening, OpeningType, WallMaterial, Room, FloorPlan, RoomType
)


@dataclass
class CementWallDetection:
    """Represents segmented regions from a raw on-site cement wall photograph."""
    wall_width_m: float
    wall_height_m: float
    material: WallMaterial
    openings: List[Opening]
    structural_keepout_zones: List[Tuple[float, float]]  # (x_start, x_end) columns/rebars
    confidence_score: float = 0.95


class VisionPerceptionPipeline:
    """
    Interface for integrating Foundation Vision Models (SAM 2, Depth Anything V2, RoomFormer).
    Converts visual inputs into metric geometric graphs.
    """

    @staticmethod
    def process_cement_wall_photo(
        photo_path: str,
        focal_length_px: Optional[float] = None,
        reference_height_m: float = 2.8,
    ) -> CementWallDetection:
        """
        Simulates / executes the Depth Anything V2 + SAM 2 pipeline on a site photo.
        1. Monocular depth estimation extracts wall surface normal and metric scale.
        2. SAM 2 segments rough-in door frames, window openings, and structural columns.
        """
        # In production with PyTorch + CUDA:
        # depth_map = depth_anything_v2.infer(photo_path)
        # masks = sam2.predict(photo_path, prompts=["concrete wall", "column", "door opening"])
        
        # Simulated robust extraction for a typical 4.5m x 2.8m concrete wall under construction:
        detected_wall = CementWallDetection(
            wall_width_m=4.8,
            wall_height_m=reference_height_m,
            material=WallMaterial.REINFORCED_CONCRETE,
            openings=[
                Opening(
                    opening_type=OpeningType.DOOR,
                    offset=0.6,
                    width=0.9,
                    height=2.1,
                    latch_side="right",
                ),
                Opening(
                    opening_type=OpeningType.WINDOW,
                    offset=2.6,
                    width=1.4,
                    height=1.2,
                ),
            ],
            structural_keepout_zones=[(4.4, 4.8)],  # Reinforced concrete shear column at corner
            confidence_score=0.96,
        )
        return detected_wall

    @staticmethod
    def build_sample_residential_floorplan() -> FloorPlan:
        """
        Creates a realistic, complete residential floor plan (Living Room, Kitchen, Bedroom, Bathroom, Corridor)
        with accurate wall dimensions, materials, and openings.
        """
        fp = FloorPlan(name="Modern Residential Apartment (2-BHK Layout)")

        # Wall Materials:
        # Exterior walls = Reinforced Concrete (thick, structural)
        # Interior wet walls = Masonry Brick
        # Bedroom partitions = Drywall

        # Corridor / Entryway
        w_corr_1 = WallSegment("W_CORR_1", Point2D(0.0, 0.0), Point2D(2.5, 0.0), material=WallMaterial.MASONRY_BRICK)
        w_corr_2 = WallSegment("W_CORR_2", Point2D(2.5, 0.0), Point2D(2.5, 2.0), material=WallMaterial.MASONRY_BRICK)
        w_corr_3 = WallSegment("W_CORR_3", Point2D(2.5, 2.0), Point2D(0.0, 2.0), material=WallMaterial.MASONRY_BRICK)
        w_corr_4 = WallSegment("W_CORR_4", Point2D(0.0, 2.0), Point2D(0.0, 0.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE,
                               openings=[Opening(OpeningType.DOOR, offset=0.5, width=1.0, latch_side="right")])
        corridor = Room("ROOM_CORR", "Entry Corridor", RoomType.CORRIDOR, [w_corr_1, w_corr_2, w_corr_3, w_corr_4])
        fp.add_room(corridor)

        # Living Room (5.0m x 4.0m)
        w_liv_1 = WallSegment("W_LIV_1", Point2D(2.5, 0.0), Point2D(7.5, 0.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE,
                              openings=[Opening(OpeningType.WINDOW, offset=2.0, width=2.0)])
        w_liv_2 = WallSegment("W_LIV_2", Point2D(7.5, 0.0), Point2D(7.5, 4.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE)
        w_liv_3 = WallSegment("W_LIV_3", Point2D(7.5, 4.0), Point2D(2.5, 4.0), material=WallMaterial.DRYWALL)
        w_liv_4 = WallSegment("W_LIV_4", Point2D(2.5, 4.0), Point2D(2.5, 0.0), material=WallMaterial.MASONRY_BRICK,
                              openings=[Opening(OpeningType.PASSAGEWAY, offset=0.5, width=1.2)])
        living = Room("ROOM_LIV", "Living & Dining Room", RoomType.LIVING_ROOM, [w_liv_1, w_liv_2, w_liv_3, w_liv_4])
        fp.add_room(living)

        # Kitchen (3.5m x 3.0m)
        w_kit_1 = WallSegment("W_KIT_1", Point2D(7.5, 0.0), Point2D(11.0, 0.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE,
                              openings=[Opening(OpeningType.WINDOW, offset=1.0, width=1.2)])
        w_kit_2 = WallSegment("W_KIT_2", Point2D(11.0, 0.0), Point2D(11.0, 3.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE)
        w_kit_3 = WallSegment("W_KIT_3", Point2D(11.0, 3.0), Point2D(7.5, 3.0), material=WallMaterial.MASONRY_BRICK)
        w_kit_4 = WallSegment("W_KIT_4", Point2D(7.5, 3.0), Point2D(7.5, 0.0), material=WallMaterial.MASONRY_BRICK,
                              openings=[Opening(OpeningType.DOOR, offset=1.8, width=0.9, latch_side="left")])
        kitchen = Room("ROOM_KIT", "Gourmet Kitchen", RoomType.KITCHEN, [w_kit_1, w_kit_2, w_kit_3, w_kit_4])
        fp.add_room(kitchen)

        # Master Bedroom (4.0m x 4.0m)
        w_bed_1 = WallSegment("W_BED_1", Point2D(2.5, 4.0), Point2D(6.5, 4.0), material=WallMaterial.DRYWALL,
                              openings=[Opening(OpeningType.DOOR, offset=0.4, width=0.9, latch_side="right")])
        w_bed_2 = WallSegment("W_BED_2", Point2D(6.5, 4.0), Point2D(6.5, 8.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE,
                              openings=[Opening(OpeningType.WINDOW, offset=1.5, width=1.8)])
        w_bed_3 = WallSegment("W_BED_3", Point2D(6.5, 8.0), Point2D(2.5, 8.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE)
        w_bed_4 = WallSegment("W_BED_4", Point2D(2.5, 8.0), Point2D(2.5, 4.0), material=WallMaterial.DRYWALL)
        bedroom = Room("ROOM_BED", "Master Bedroom", RoomType.BEDROOM, [w_bed_1, w_bed_2, w_bed_3, w_bed_4])
        fp.add_room(bedroom)

        # Master Bathroom (2.5m x 2.2m)
        w_bath_1 = WallSegment("W_BATH_1", Point2D(0.0, 2.0), Point2D(2.5, 2.0), material=WallMaterial.MASONRY_BRICK,
                               openings=[Opening(OpeningType.DOOR, offset=0.3, width=0.8, latch_side="right")])
        w_bath_2 = WallSegment("W_BATH_2", Point2D(2.5, 2.0), Point2D(2.5, 4.2), material=WallMaterial.MASONRY_BRICK)
        w_bath_3 = WallSegment("W_BATH_3", Point2D(2.5, 4.2), Point2D(0.0, 4.2), material=WallMaterial.MASONRY_BRICK)
        w_bath_4 = WallSegment("W_BATH_4", Point2D(0.0, 4.2), Point2D(0.0, 2.0), is_exterior=True, material=WallMaterial.REINFORCED_CONCRETE)
        bathroom = Room("ROOM_BATH", "Master Bathroom", RoomType.BATHROOM, [w_bath_1, w_bath_2, w_bath_3, w_bath_4])
        fp.add_room(bathroom)

        # Set panel location near entry corridor exterior wall
        fp.panel_location = Point2D(0.2, 0.4)

        return fp

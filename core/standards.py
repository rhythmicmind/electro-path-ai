"""
Electrical Code Standards & Regulatory Constraints (NEC / IEC 60364 compliant).
Defines dimensional spacing, fixture ratings, circuit limits, and safety regulations.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class FixtureType(Enum):
    MAIN_PANEL = "main_panel"               # Main Distribution Board (MDB)
    DUPLEX_15A = "duplex_15a"               # Standard general convenience receptacle
    GFCI_20A = "gfci_20a"                   # Ground-Fault Circuit-Interrupter (wet zones)
    DEDICATED_20A = "dedicated_20a"         # Dedicated appliance (fridge, microwave, washer)
    HEAVY_DUTY_240V = "heavy_duty_240v"     # Electric range, oven, dryer, HVAC compressor
    SWITCH_SINGLE_POLE = "switch_1p"        # Standard wall switch
    SWITCH_3WAY = "switch_3way"             # Hallway/stairway multi-location switch
    CEILING_LIGHT = "ceiling_light"         # Luminaire / ceiling fixture
    PULL_BOX = "pull_box"                   # Junction / pull box inserted for conduit bend relief


@dataclass
class FixtureSpec:
    fixture_type: FixtureType
    voltage: float                          # Volts (120V / 240V / 230V)
    amperage: float                         # Rated Amps
    typical_load_watts: float               # Estimated continuous/design load (Watts)
    height_above_floor_m: float             # Standard mounting height AFF (meters)
    requires_gfci: bool = False
    is_dedicated: bool = False


# Standard specifications table
FIXTURE_SPECS: Dict[FixtureType, FixtureSpec] = {
    FixtureType.MAIN_PANEL: FixtureSpec(
        fixture_type=FixtureType.MAIN_PANEL,
        voltage=240.0,
        amperage=200.0,
        typical_load_watts=0.0,
        height_above_floor_m=1.6,
    ),
    FixtureType.DUPLEX_15A: FixtureSpec(
        fixture_type=FixtureType.DUPLEX_15A,
        voltage=120.0,
        amperage=15.0,
        typical_load_watts=180.0,           # NEC standard receptacle load rating
        height_above_floor_m=0.30,          # 12 inches AFF
        requires_gfci=False,
    ),
    FixtureType.GFCI_20A: FixtureSpec(
        fixture_type=FixtureType.GFCI_20A,
        voltage=120.0,
        amperage=20.0,
        typical_load_watts=240.0,
        height_above_floor_m=1.10,          # 44 inches AFF (above counters / basins)
        requires_gfci=True,
    ),
    FixtureType.DEDICATED_20A: FixtureSpec(
        fixture_type=FixtureType.DEDICATED_20A,
        voltage=120.0,
        amperage=20.0,
        typical_load_watts=1500.0,          # Dedicated high draw (microwave, fridge)
        height_above_floor_m=0.30,
        is_dedicated=True,
    ),
    FixtureType.HEAVY_DUTY_240V: FixtureSpec(
        fixture_type=FixtureType.HEAVY_DUTY_240V,
        voltage=240.0,
        amperage=50.0,
        typical_load_watts=8000.0,          # Electric range / induction cooktop
        height_above_floor_m=0.30,
        is_dedicated=True,
    ),
    FixtureType.SWITCH_SINGLE_POLE: FixtureSpec(
        fixture_type=FixtureType.SWITCH_SINGLE_POLE,
        voltage=120.0,
        amperage=15.0,
        typical_load_watts=15.0,            # Control load
        height_above_floor_m=1.20,          # 48 inches AFF
    ),
    FixtureType.CEILING_LIGHT: FixtureSpec(
        fixture_type=FixtureType.CEILING_LIGHT,
        voltage=120.0,
        amperage=15.0,
        typical_load_watts=60.0,            # LED fixture allowance
        height_above_floor_m=2.80,          # Ceiling flush
    ),
    FixtureType.PULL_BOX: FixtureSpec(
        fixture_type=FixtureType.PULL_BOX,
        voltage=0.0,
        amperage=0.0,
        typical_load_watts=0.0,
        height_above_floor_m=2.60,
    ),
}


class ElectricalStandards:
    """
    Encapsulates NEC (National Electrical Code) and IEC dimensional and circuit safety constraints.
    """
    # Spacing Constraints (NEC 210.52)
    GENERAL_WALL_MIN_LENGTH_M = 0.60         # Walls >= 2 ft require receptacles
    GENERAL_MAX_SPACING_M = 3.65             # Max 12 ft between outlets
    GENERAL_CORNER_MAX_DIST_M = 1.80         # Max 6 ft from door/corner to first outlet
    
    KITCHEN_COUNTER_MAX_SPACING_M = 1.20     # Max 4 ft along counter
    KITCHEN_COUNTER_EDGE_DIST_M = 0.60       # Max 2 ft from counter edge
    BATHROOM_BASIN_MAX_DIST_M = 0.90         # Max 3 ft from edge of basin
    
    DOOR_SWITCH_JAMB_OFFSET_M = 0.20         # 8 inches from latch jamb
    
    # Circuit Branch Capacity Limits (80% Continuous Load Rule - NEC 210.19)
    MAX_CONTINUOUS_LOAD_15A_WATTS = 1440.0   # 120V * 15A * 0.8
    MAX_CONTINUOUS_LOAD_20A_WATTS = 1920.0   # 120V * 20A * 0.8
    
    # Conduit Installation Regulations (NEC 358.26 / Chapter 9)
    MAX_CONDUIT_BENDS_DEGREES = 360.0        # Max 4 quarter bends (360 deg) between pull boxes
    MAX_VOLTAGE_DROP_PERCENT = 3.0           # Recommended max voltage drop for branch circuits
    
    # Wire Resistances (Ohms per meter of copper conductor at 75°C)
    COPPER_RESISTANCE_OHMS_PER_M = {
        14: 0.0084,  # 14 AWG (15A branch)
        12: 0.0052,  # 12 AWG (20A branch)
        10: 0.0033,  # 10 AWG (30A branch)
        6:  0.0013,  # 6 AWG (50A branch)
    }

    @classmethod
    def calculate_voltage_drop(cls, current_amps: float, length_meters: float, wire_awg: int, voltage: float = 120.0) -> float:
        """
        Calculates percentage voltage drop: V_drop = 2 * I * R * L / V * 100%
        """
        r_per_m = cls.COPPER_RESISTANCE_OHMS_PER_M.get(wire_awg, 0.0052)
        # Round-trip length = 2 * length_meters
        v_drop = 2.0 * current_amps * (r_per_m * length_meters)
        return (v_drop / voltage) * 100.0

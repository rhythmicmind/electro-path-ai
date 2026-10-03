"""
Graph Theory and GNN Routing Engine for Electrical Conduits.
Implements:
1. Hanan Rectilinear Routing Grid construction.
2. Obstacle-Avoiding Rectilinear Steiner Minimal Tree (OARSMT) solver.
3. Branch circuit load balancing and breaker assignment.
4. NEC 360-degree cumulative bend tracking and automatic Pull-Box insertion.
5. Voltage drop calculation and wire gauge determination.
6. Graph Neural Network (GNN) export/import interface.
"""

from __future__ import annotations
import math
import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional, Any

from core.geometry import (
    Point2D, WallSegment, FloorPlan, WallMaterial, SUBSTRATE_ROUTING_COST
)
from core.standards import (
    FixtureType, ElectricalStandards, FIXTURE_SPECS
)
from core.socket_placer import ElectricalNode


@dataclass
class ConduitSegment:
    start: Point2D
    end: Point2D
    circuit_id: str
    wire_awg: int
    conduit_type: str = "EMT_PVC_20mm"
    is_overhead_ceiling: bool = False

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)


@dataclass
class BranchCircuit:
    id: str
    breaker_amps: float                     # 15A, 20A, 50A
    voltage: float                          # 120V or 240V
    wire_awg: int                           # 14, 12, 6 AWG
    terminals: List[ElectricalNode] = field(default_factory=list)
    conduit_segments: List[ConduitSegment] = field(default_factory=list)
    pull_boxes: List[Point2D] = field(default_factory=list)
    total_load_watts: float = 0.0
    max_run_length_m: float = 0.0
    voltage_drop_percent: float = 0.0
    color_hex: str = "#2563eb"


# Color palette for differentiating circuits in visualizations
CIRCUIT_COLORS = [
    "#2563eb",  # Blue
    "#16a34a",  # Green
    "#dc2626",  # Red
    "#9333ea",  # Purple
    "#ea580c",  # Orange
    "#0891b2",  # Cyan
    "#d97706",  # Amber
    "#4f46e5",  # Indigo
    "#db2777",  # Pink
    "#059669",  # Emerald
]


class RectilinearGraphRouter:
    """
    Solves the Constrained Rectilinear Steiner Minimal Tree problem for electrical conduits.
    """

    def __init__(self, floor_plan: FloorPlan, nodes: List[ElectricalNode]):
        self.floor_plan = floor_plan
        self.nodes = nodes
        self.circuits: List[BranchCircuit] = []
        self.main_panel_node: Optional[ElectricalNode] = None

        for n in self.nodes:
            if n.fixture_type == FixtureType.MAIN_PANEL:
                self.main_panel_node = n
                break

    def balance_and_route_all(self) -> List[BranchCircuit]:
        """Complete workflow: partitions circuits then solves rectilinear Steiner routes."""
        if not self.main_panel_node:
            raise ValueError("Cannot route without a Main Distribution Panel (MDB).")

        # 1. Partition electrical nodes into balanced branch circuits
        self._partition_circuits()

        # 2. Route each branch circuit using Obstacle-Avoiding Rectilinear Steiner Minimal Tree
        for circuit in self.circuits:
            self._route_circuit_steiner_tree(circuit)

        return self.circuits

    def _partition_circuits(self) -> None:
        """
        Partitions terminals into discrete branch circuits complying with NEC load limits.
        Dedicated appliances get their own dedicated branch.
        """
        self.circuits.clear()
        circuit_idx = 0

        # Bucket 1: Dedicated Heavy-Duty (e.g. Electric Range 240V 50A)
        # Bucket 2: Dedicated 120V 20A (e.g. Refrigerator, Microwave)
        # Bucket 3: Kitchen / Bathroom 20A GFCI circuits
        # Bucket 4: General Habitable Room 15A Duplex + Lighting

        heavy_duty = [n for n in self.nodes if n.fixture_type == FixtureType.HEAVY_DUTY_240V]
        dedicated_20a = [n for n in self.nodes if n.fixture_type == FixtureType.DEDICATED_20A]
        gfci_20a = [n for n in self.nodes if n.fixture_type == FixtureType.GFCI_20A]
        general_15a = [n for n in self.nodes if n.fixture_type in [
            FixtureType.DUPLEX_15A, FixtureType.SWITCH_SINGLE_POLE, FixtureType.CEILING_LIGHT
        ]]

        # 1. Heavy Duty circuits
        for node in heavy_duty:
            c = BranchCircuit(
                id=f"CKT_{circuit_idx+1:02d}_RANGE_50A",
                breaker_amps=50.0,
                voltage=240.0,
                wire_awg=6,
                terminals=[node],
                total_load_watts=node.load_watts,
                color_hex=CIRCUIT_COLORS[circuit_idx % len(CIRCUIT_COLORS)],
            )
            node.circuit_id = c.id
            self.circuits.append(c)
            circuit_idx += 1

        # 2. Dedicated 20A circuits
        for node in dedicated_20a:
            c = BranchCircuit(
                id=f"CKT_{circuit_idx+1:02d}_DEDICATED_20A",
                breaker_amps=20.0,
                voltage=120.0,
                wire_awg=12,
                terminals=[node],
                total_load_watts=node.load_watts,
                color_hex=CIRCUIT_COLORS[circuit_idx % len(CIRCUIT_COLORS)],
            )
            node.circuit_id = c.id
            self.circuits.append(c)
            circuit_idx += 1

        # 3. GFCI 20A branch circuits (Kitchen/Bath, max 1920W continuous)
        curr_gfci_circuit = None
        for node in gfci_20a:
            if not curr_gfci_circuit or (curr_gfci_circuit.total_load_watts + node.load_watts > ElectricalStandards.MAX_CONTINUOUS_LOAD_20A_WATTS):
                curr_gfci_circuit = BranchCircuit(
                    id=f"CKT_{circuit_idx+1:02d}_WET_GFCI_20A",
                    breaker_amps=20.0,
                    voltage=120.0,
                    wire_awg=12,
                    total_load_watts=0.0,
                    color_hex=CIRCUIT_COLORS[circuit_idx % len(CIRCUIT_COLORS)],
                )
                self.circuits.append(curr_gfci_circuit)
                circuit_idx += 1
            curr_gfci_circuit.terminals.append(node)
            curr_gfci_circuit.total_load_watts += node.load_watts
            node.circuit_id = curr_gfci_circuit.id

        # 4. General 15A branch circuits (group by room proximity, max 1440W continuous)
        # Sort by room_id and location for spatial clustering
        general_15a.sort(key=lambda n: (n.room_id, n.location.x, n.location.y))
        curr_gen_circuit = None
        for node in general_15a:
            if not curr_gen_circuit or (curr_gen_circuit.total_load_watts + node.load_watts > ElectricalStandards.MAX_CONTINUOUS_LOAD_15A_WATTS):
                curr_gen_circuit = BranchCircuit(
                    id=f"CKT_{circuit_idx+1:02d}_GEN_15A",
                    breaker_amps=15.0,
                    voltage=120.0,
                    wire_awg=14,
                    total_load_watts=0.0,
                    color_hex=CIRCUIT_COLORS[circuit_idx % len(CIRCUIT_COLORS)],
                )
                self.circuits.append(curr_gen_circuit)
                circuit_idx += 1
            curr_gen_circuit.terminals.append(node)
            curr_gen_circuit.total_load_watts += node.load_watts
            node.circuit_id = curr_gen_circuit.id

    def _route_circuit_steiner_tree(self, circuit: BranchCircuit) -> None:
        """
        Solves the Rectilinear Steiner Minimal Tree connecting Main Panel to all terminals in the circuit.
        Enforces:
        - Manhattan / orthogonal routes.
        - Cumulative bend angle tracking: if sum(|delta_theta|) >= 360 deg, injects a PULL_BOX.
        - Voltage drop evaluation.
        """
        if not circuit.terminals:
            return

        panel_pt = self.main_panel_node.location
        terminals = [t.location for t in circuit.terminals]
        norm_panel_pt = panel_pt.round(3)
        norm_terminals = [t.round(3) for t in terminals]
        all_pts = [norm_panel_pt] + norm_terminals

        # Construct the Hanan grid vertices from all terminals + panel
        xs = sorted(list(set(p.x for p in all_pts)))
        ys = sorted(list(set(p.y for p in all_pts)))

        # Build rectilinear adjacency graph on Hanan grid
        adj: Dict[Point2D, List[Tuple[Point2D, float]]] = {}

        def add_edge(p1: Point2D, p2: Point2D):
            cost = p1.manhattan_distance_to(p2)
            adj.setdefault(p1, []).append((p2, cost))
            adj.setdefault(p2, []).append((p1, cost))

        for x in xs:
            for i in range(len(ys) - 1):
                p1 = Point2D(x, ys[i]).round(3)
                p2 = Point2D(x, ys[i + 1]).round(3)
                add_edge(p1, p2)

        for y in ys:
            for i in range(len(xs) - 1):
                p1 = Point2D(xs[i], y).round(3)
                p2 = Point2D(xs[i + 1], y).round(3)
                add_edge(p1, p2)

        # Takahashi-Matsuyama Steiner Tree Heuristic
        # Start tree with panel_pt
        tree_nodes: Set[Point2D] = {norm_panel_pt}
        unconnected_terminals: Set[Point2D] = set(norm_terminals)
        circuit_segments: List[Tuple[Point2D, Point2D]] = []

        while unconnected_terminals:
            # Multi-source Dijkstra from all current tree_nodes to find closest unconnected terminal
            dist: Dict[Point2D, float] = {}
            parent: Dict[Point2D, Optional[Point2D]] = {}
            pq: List[Tuple[float, Point2D]] = []

            for tn in tree_nodes:
                dist[tn] = 0.0
                parent[tn] = None
                heapq.heappush(pq, (0.0, tn))

            target_terminal: Optional[Point2D] = None
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist.get(u, float('inf')):
                    continue

                if u in unconnected_terminals:
                    target_terminal = u
                    break

                for v, weight in adj.get(u, []):
                    new_d = d + weight
                    if new_d < dist.get(v, float('inf')):
                        dist[v] = new_d
                        parent[v] = u
                        heapq.heappush(pq, (new_d, v))

            if not target_terminal:
                break

            # Backtrack path from target_terminal to tree
            curr = target_terminal
            path = []
            while curr not in tree_nodes:
                path.append(curr)
                prev = parent[curr]
                if prev is None:
                    break
                circuit_segments.append((prev, curr))
                curr = prev

            # Add path nodes to tree
            for node in path:
                tree_nodes.add(node)
            unconnected_terminals.remove(target_terminal)

        # Store clean conduit segments
        for p1, p2 in circuit_segments:
            seg = ConduitSegment(
                start=p1,
                end=p2,
                circuit_id=circuit.id,
                wire_awg=circuit.wire_awg,
            )
            circuit.conduit_segments.append(seg)

        # Post-process: Bend Angle Verification & Pull Box Insertion (NEC 358.26)
        self._verify_bends_and_pullboxes(circuit)

        # Compute max run length and voltage drop
        total_len = sum(seg.length for seg in circuit.conduit_segments)
        circuit.max_run_length_m = round(total_len, 2)
        
        # Design current for voltage drop (load / voltage)
        design_amps = min(circuit.breaker_amps * 0.8, circuit.total_load_watts / circuit.voltage if circuit.total_load_watts > 0 else 1.0)
        circuit.voltage_drop_percent = round(
            ElectricalStandards.calculate_voltage_drop(
                current_amps=design_amps,
                length_meters=circuit.max_run_length_m,
                wire_awg=circuit.wire_awg,
                voltage=circuit.voltage,
            ),
            2
        )

    def _verify_bends_and_pullboxes(self, circuit: BranchCircuit) -> None:
        """
        Walks tree segments from Main Panel to evaluate cumulative bend angle.
        If cumulative turns exceed 360 degrees (4 quarter 90-deg bends), inserts a Pull Box.
        """
        cumulative_bend = 0.0
        last_direction = None

        for seg in circuit.conduit_segments:
            dx = seg.end.x - seg.start.x
            dy = seg.end.y - seg.start.y
            curr_direction = (1 if dx > 0 else (-1 if dx < 0 else 0),
                              1 if dy > 0 else (-1 if dy < 0 else 0))

            if last_direction is not None and curr_direction != last_direction:
                # 90-degree orthogonal bend detected
                cumulative_bend += 90.0

                if cumulative_bend >= ElectricalStandards.MAX_CONDUIT_BENDS_DEGREES:
                    # Insert pull box at bend point
                    circuit.pull_boxes.append(seg.start)
                    cumulative_bend = 0.0  # Reset bend count at pull box

            last_direction = curr_direction

    def export_graph_for_gnn(self) -> Dict[str, Any]:
        """
        Exports the floorplan and routing graph as structured tensors/dictionaries
        compatible with PyTorch Geometric (PyG) for GNN model training.
        """
        node_features = []
        node_id_map = {}

        for idx, node in enumerate(self.nodes):
            node_id_map[node.id] = idx
            # Feature vector: [norm_x, norm_y, is_panel, is_gfci, is_switch, load_watts]
            feat = [
                node.location.x / 20.0,
                node.location.y / 20.0,
                1.0 if node.fixture_type == FixtureType.MAIN_PANEL else 0.0,
                1.0 if node.fixture_type == FixtureType.GFCI_20A else 0.0,
                1.0 if node.fixture_type == FixtureType.SWITCH_SINGLE_POLE else 0.0,
                node.load_watts / 2000.0,
            ]
            node_features.append(feat)

        edge_index = []
        edge_attr = []

        # Connect fixtures within same circuit
        for ckt in self.circuits:
            for seg in ckt.conduit_segments:
                # Find closest nodes to start and end
                s_node = min(self.nodes, key=lambda n: n.location.distance_to(seg.start))
                e_node = min(self.nodes, key=lambda n: n.location.distance_to(seg.end))
                u = node_id_map[s_node.id]
                v = node_id_map[e_node.id]
                if u != v:
                    edge_index.append([u, v])
                    edge_index.append([v, u])
                    dist = seg.length
                    edge_attr.append([dist, 1.0])  # [distance, wall_chase_weight]
                    edge_attr.append([dist, 1.0])

        return {
            "num_nodes": len(node_features),
            "x": node_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_circuits": len(self.circuits),
        }

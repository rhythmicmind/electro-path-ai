# ElectroPath AI: Intelligent Electrical Layout & Conduit Routing

> **Automated, NEC-compliant socket placement and rectilinear conduit routing from architectural floor plans and rough-in cement wall photographs.**

---

## ⚡ Overview

ElectroPath AI bridges **Computer Vision**, **Statutory Building Codes (NEC/IEC)**, and **Graph Theory / Graph Neural Networks (GNNs)** to automate residential and commercial electrical planning:

1. **Perception**: Ingests architectural CAD/PDF blueprints (via RoomFormer / CubiCasa5k vectorization) or on-site cement wall photographs (via Depth Anything V2 metric depth and SAM 2 segmentation).
2. **Code Compliance Engine**: Automatically samples sockets, switches, luminaires, and the Main Distribution Board (MDB) satisfying NEC 210.52 6ft/12ft rules, counter GFCI regulations, dedicated heavy-appliance circuits, and door-switch ergonomics.
3. **Graph Theory & GNN Router**:
   - Constructs a rectilinear **Hanan Routing Grid** along walls and overhead ceiling spaces.
   - Solves the **Obstacle-Avoiding Rectilinear Steiner Minimal Tree (OARSMT)**.
   - Enforces physical wiring constraints: **cumulative bend tracking** with automatic **Pull Box insertion** at $\ge 360^\circ$ turns (NEC 358.26).
   - Validates **voltage drop** ($< 3\%$) and balances branch circuits.
4. **Visual Dashboard & BOM**: Produces interactive HTML5/SVG layouts, breaker schedules, and a comprehensive Bill of Materials (conduit meters, wire gauge breakdown, device counts).

---

## 📁 Repository Structure

```
electro-path-ai/
├── core/
│   ├── geometry.py             # 2D/3D points, walls, openings, rooms, and substrate chasing costs
│   ├── standards.py            # NEC/IEC electrical code rules, fixture specs, voltage drop
│   ├── socket_placer.py        # Automated code-compliant fixture placement engine
│   ├── graph_router.py         # Hanan grid, Steiner Minimal Tree, bend angle tracking, GNN export
│   └── vision_pipeline.py      # Blueprint parser & Depth/SAM2 cement wall photo interface
├── visualizer/
│   └── report_generator.py     # Standalone interactive SVG/HTML layout and BOM dashboard
├── output/
│   ├── electrical_layout_plan.html # Interactive visual inspection dashboard
│   └── gnn_training_graph.json     # PyTorch Geometric ready graph training dataset
├── main.py                     # Main execution and benchmark pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Requirements & Installation
Python 3.9+ is required.

```bash
cd electro-path-ai
pip install -r requirements.txt
```

*(Core routing algorithms are self-contained in pure Python with zero mandatory dependencies, but `networkx`, `numpy`, and `scipy` are supported for extended graph operations).*

### 2. Run the Full Optimization Pipeline

```bash
python main.py
```

### 3. Open the Interactive Visualizer

After running, open the generated dashboard in your browser:
```text
file:///C:/Users/sriad/.gemini/antigravity-ide/scratch/electro-path-ai/output/electrical_layout_plan.html
```

---

## 🧠 Graph Theory & GNN Architecture

- **Hanan Grid Theorem**: In rectilinear routing, an optimal Steiner tree lies entirely on the orthogonal grid formed by horizontal and vertical lines passing through terminal points.
- **Edge Weighting via GNN (GATv2)**: Concrete requires costly wall grooving (cost multiplier $\approx 4.5\times$), while drywall is light ($\approx 1.0\times$) and ceiling trays are favored ($\approx 0.5\times$). GNN edge embeddings predict wall drillability from visual features.
- **Pull-Box Insertion**: When $\sum |\Delta\theta| \ge 360^\circ$ on any run, the router automatically inserts an intermediate junction box to avoid cable jamming during wire pulling.
- **GNN Graph Export**: Run outputs `output/gnn_training_graph.json` containing node features, directed edge indices, and substrate attributes for PyTorch Geometric training.

---

## 📚 Recommended Open-Source Datasets

- **Floorplans**: [CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k) & [FloorPlanCAD](https://github.com/woodfrog/FloorPlanCAD)
- **Indoor 3D / Cement Scans**: [Structured3D](https://structured3d-dataset.org/) & [Matterport3D](https://niessner.github.io/Matterport/)
- **Graph Datasets**: [R-Plan](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) & [HouseGAN++](https://github.com/nelson-nauata/HouseGAN)
- **Pretrained Foundation Models**: `Depth Anything V2 (Metric Indoor)` & `SAM 2 (Meta AI)`

---

## 📄 License
MIT License

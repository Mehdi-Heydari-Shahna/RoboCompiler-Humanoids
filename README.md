# RoboCompiler-Humanoids

**Composable Dynamics for Cooperative Humanoids: A RoboCompiler Extension**

Mehdi Heydari Shahna · Jouni Mattila

Preprint: [arXiv:2610.14237](https://arxiv.org/abs/2610.14237)

This repository extends [RoboCompiler](https://github.com/Mehdi-Heydari-Shahna/RoboCompiler) to physically connected humanoids. Member-wise dynamics are compiled from physical records (rigid bodies, moving actuator parts, and reflected rotor inertias) with analytic constraint curvature, and a two-level composition keeps each robot's model reusable while ground supports and hand connections change. Five hand-linked closed-chain humanoids are simulated in ring, semicircle, and line formations.

The repository contains the code, the result files of the reported runs, the manuscript, and the videos.

## Main capabilities

- Member-level inertia, Coriolis, gravity, damping, and actuation terms assembled directly from compiled physical records, with analytic lift curvature and a member-wise power identity for floating-base quasi-velocities.
- Two-level composition that retains each robot's internal reduction across support and connection changes.
- Local inertial updates of rank at most six, and prescribed-acceleration constraint-wrench maps that are affine in the actuator efforts.
- One physical record, three views: the assembled dynamics model, a MuJoCo (MJCF) model, and a Pinocchio model with rigid constraint models.
- Actuator-aware tracking, coupled force allocation, payload updates, and support-mode switching for five hand-linked humanoids.

## Repository layout

| Path | Content |
|---|---|
| `ipd/` | Graph compilation, closure modes with analytic lift rates, inertia-partitioned dynamics, two-level composition, local updates and constraint wrenches, controller, extended-precision reference, and MuJoCo and Pinocchio backends |
| `dancers/` | Closed-chain humanoid record, formations, choreography, side stepping, plants, and closed-loop execution |
| `examples/` | The three-degree-of-freedom series–parallel manipulator of arXiv:2510.23226 and a loader for an optional excavator record supplied by the user (not distributed) |
| `vendor/` | The PACDM solver of the RoboCompiler repository, unchanged |
| `workflows/` | Studies (`dance/`, `statewise/`, `manipulator/`) and report generators (`report/`) |
| `tests/` | Unit tests |
| `results/` | Result files of the reported runs (JSON summaries and NumPy logs) |
| `paper/` | Manuscript |
| `videos/` | Videos of the closed-loop runs |

## Getting started

The results were produced with **Python 3.12 on Linux**. `requirements.txt` pins the numerical packages to the versions recorded in the result files.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Pinocchio's PyPI package (`pin`) has no Windows wheels; on Windows, use WSL.

| Command | Purpose |
|---|---|
| `python reproduce.py --profile smoke` | Short runs of every workflow except the dance audit (a few minutes) |
| `python reproduce.py --profile full` | Every run reported in the paper (about one hour on two cores) |
| `python reproduce.py --report` | Tables, numbers, and figures from `results/` (two figures are rendered: needs OpenGL, e.g. `MUJOCO_GL=egl`) |
| `python reproduce.py --report --videos` | Also the videos (needs OpenGL, e.g. `MUJOCO_GL=egl`) |
| `python -m pytest -q tests` | Unit tests (after `python -m pip install pytest`) |

The excavator cases of the state-wise studies need an excavator record, which is not distributed with this repository. Set `ROBOCOMPILER_EXCAVATOR_RECORD=/path/to/record.json` (schema `cmg.physical-closed-chain/1.0` or `cmg.physical-mechanism-graph/1.0`, with partition requests `joint_space` and/or `cylinder_space`), or pass `--excavator-record PATH` to `reproduce.py` or `workflows/statewise/run_statewise.py`. Without a record, these cases and `tests/test_repository_records.py` are skipped, and the shipped results do not contain them.

The full profile writes to `results/` unless `--out` is given. `--report` writes `paper/generated/`, `paper/figures/`, and `evidence/`; `python verify_evidence.py` then checks the evidence files against the hash of the result-producing sources. Every result file records that hash, the package versions, and the time of writing (`ipd/provenance.py`).

## Demonstration videos

Five closed-chain humanoids with the partitioned model on the realistic MuJoCo plant.

| Formation | Dance | Side steps |
|---|---|---|
| Ring | [Video](videos/ring_dance.mp4) | [Video](videos/ring_steps.mp4) |
| Semicircle | [Video](videos/semicircle_dance.mp4) | [Video](videos/semicircle_steps.mp4) |
| Line | [Video](videos/line_dance.mp4) | [Video](videos/line_steps.mp4) |

During the side steps, every lift-off and landing switches the composer, and the five robot modules are reused. [Per-robot models vs. composed model](videos/ring_steps_per_robot_vs_composed.mp4) shows the travelling ring with five separate robot models (hand links not modelled) and with the composed partitioned model.

## Scope

The humanoid is a synthetic closed-chain design (42 bodies, 49 joints, 21 actuator ports) with representative drive data of electric humanoids; the five humanoids differ in link dimensions and inertias. Hand holding is modelled by ideal spherical joints between rigid hand links. The closed-loop studies run on MuJoCo and Pinocchio plants.

## Paper and citation

Please cite the accompanying manuscript when using this work:

Mehdi Heydari Shahna and Jouni Mattila. **Composable Dynamics for Cooperative Humanoids: A RoboCompiler Extension.** arXiv:2610.14237, 2026. https://arxiv.org/abs/2610.14237. [Manuscript](paper/robocompiler_humanoids.pdf).

This work builds on RoboCompiler: Mehdi Heydari Shahna, Joongheon Kim, and Jouni Mattila. **RoboCompiler: Graph-Native Compilation of Closed-Chain Robots for Consistent Modeling, Control, and Simulation.** 2026. https://doi.org/10.48550/arXiv.2609.35717

## License

Apache License 2.0; see [LICENSE](LICENSE).

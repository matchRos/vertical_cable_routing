# vertical_cable_routing

Multi-package ROS repository for cable routing, manipulation, planning, and operator tooling.

This repository is the new home for the current `debug_gui` workflow and the related cable-routing logic. The goal is to end up with a self-contained codebase that:

- owns its own planning, perception, GUI, and orchestration code
- talks to robot drivers and third-party systems through ROS interfaces
- does not depend on imports from the legacy `cable_routing` Python package
- can be built and run without Conda

## Package Layout

- `cable_msgs`
  Shared ROS `msg`, `srv`, and `action` definitions.
- `cable_core`
  Reusable geometry, board, clip, config, and routing primitives with minimal ROS coupling.
- `cable_perception`
  Camera access, image handling, tracing, and projection services.
- `cable_planning`
  Grasp planning, clip routing targets, and route planning services/actions.
- `cable_motion`
  Motion commands, execution clients, and motion-status handling for robot actions.
- `cable_orchestrator`
  Pipeline execution, state machine logic, and high-level task orchestration.
- `cable_studio`
  Operator and development GUI. This replaces the old `debug_gui` naming.
- `cable_bringup`
  Launch files and system-level configuration for wiring the full stack together.
- `cable_routing_utilities`
  Existing calibration utilities kept during migration.

## Migration Principles

- Keep pure computation in Python libraries instead of turning every helper into a ROS node.
- Use ROS actions/services/topics at process boundaries and hardware boundaries.
- Migrate incrementally so the stack stays runnable throughout the move.
- Prefer explicit package ownership over large shared utility buckets.

## Build Strategy

This repository is intended to be used as a multi-package source repository inside a catkin workspace. Example:

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone <repo-url> vertical_cable_routing
cd ..
rosdep install --from-paths src --ignore-src -r -y
catkin build
```

## Getting Rid Of Conda

The target setup is:

- ROS dependencies from `package.xml` and `rosdep`
- Python dependencies installed either from Ubuntu/ROS packages or a small `pip` requirements file
- no mandatory `conda activate ...` step

The current Conda environment is still useful as a dependency inventory, but it should not remain the runtime contract for this repository.

See `docs/architecture.md` and `docs/migration_plan.md` for the detailed target structure and migration order.

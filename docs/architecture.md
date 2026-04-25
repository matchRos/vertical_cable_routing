# Architecture

## Goals

- Self-contained cable-routing repository
- Clean separation between computation, ROS interfaces, orchestration, and GUI
- Replace legacy `debug_gui` naming with a product/development-facing structure
- Remove implicit imports from the old `cable_routing` package
- Remove the Conda runtime requirement

## Package Responsibilities

### `cable_msgs`

Shared ROS message, service, and action definitions.

Expected examples:

- motion status
- step result
- route target plans
- tracing requests/results

### `cable_core`

Pure Python logic and data structures with minimal ROS coupling.

Expected contents:

- board and clip geometry
- projection helpers
- routing primitives for U-clips, C-clips, pegs
- config loading and validation
- general math and transform helpers

### `cable_perception`

ROS-facing perception components.

Expected contents:

- camera subscribers/clients
- tracing services
- image preprocessing
- pixel-to-world conversion services

### `cable_planning`

Planning modules that produce actionable targets but do not directly execute robot motion.

Expected contents:

- grasp planning
- first-route planning
- clip-specific route planners
- path shaping and pose target generation

### `cable_motion`

ROS integration for robot motion and gripper execution.

Expected contents:

- MoveIt goal clients
- Cartesian motion clients
- motion completion/status waiting
- gripper command wrappers

### `cable_orchestrator`

High-level control flow for complete cable-routing tasks.

Expected contents:

- pipeline runner
- action orchestration
- state machine
- recovery and retry policy

### `cable_studio`

GUI for development, operation, visualization, and manual control.

Expected contents:

- Qt windowing
- step/action result display
- overlays and operator feedback
- manual execution tools

### `cable_bringup`

System composition.

Expected contents:

- launch files
- default runtime configs
- profiles for sim/dev/real hardware

## Design Rules

- Keep non-ROS computation in `cable_core` even if it is called from ROS nodes.
- Let `cable_motion`, `cable_perception`, and `cable_planning` expose ROS-facing contracts.
- Keep orchestration decisions centralized in `cable_orchestrator`.
- Keep GUI code in `cable_studio`; avoid putting planning or motion logic directly into the UI.

## No-Conda Direction

Target dependency flow:

1. System ROS installation
2. `rosdep` for ROS and Ubuntu package dependencies
3. A very small `pip` surface only where necessary

Conda can still be used temporarily as a migration aid, but new package definitions should target `package.xml` and `rosdep` first.

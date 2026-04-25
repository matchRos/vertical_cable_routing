# Migration Plan

## Current External Couplings

The existing `debug_gui` stack is already mostly local. The main remaining imports from the legacy repository are:

- `cable_routing.env.ext_camera.utils.img_utils`
- `cable_routing.env.ext_camera.ros.zed_camera`
- `cable_routing.env.robots.misc.calculate_sequence`
- `cable_routing.handloom.handloom_pipeline.single_tracer`

This is small enough to migrate incrementally.

## Phase 1: Skeleton And Naming

- Create the new package structure
- Introduce `cable_studio` as the replacement for `debug_gui`
- Document the package boundaries and migration order

## Phase 2: Core Extraction

Move the least ROS-coupled modules first into `cable_core`:

- board geometry and debug board models
- clip type definitions
- board projection helpers
- routing planes
- motion primitives for clip geometry
- config loading helpers

## Phase 3: Planning Extraction

Move planning-specific modules into `cable_planning`:

- grasp pose service
- first-route target service
- clip target service
- handover pose service

At this phase we should replace the dependency on `calculate_sequence` with a local implementation.

## Phase 4: Perception Extraction

Move perception and trace-related modules into `cable_perception`:

- tracing service
- path projection service
- cable orientation service
- camera integration wrappers

At this phase we should replace imports from `img_utils`, `zed_camera`, and `single_tracer`.

## Phase 5: Motion Extraction

Move motion-specific wrappers into `cable_motion`:

- arm motion waiters
- robot command wrappers
- gripper helpers
- status normalization

## Phase 6: Orchestrator And Studio

Move the current pipeline, orchestrator, and GUI into:

- `cable_orchestrator`
- `cable_studio`

This phase should also rename modules and user-facing labels away from `debug_*`.

## Phase 7: Bringup And Runtime Cleanup

- Add launch files for the full stack
- define runtime config sets
- remove legacy repository imports
- verify that Conda is no longer required

## Conda Removal Checklist

- Convert Python package requirements into `package.xml` dependencies where possible
- Add `rosdep` keys for ROS-facing dependencies
- Identify the small set of true pip-only dependencies
- Create a minimal install recipe that works with system ROS plus `catkin build`

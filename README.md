# Robotics II – Intelligent Robotic Systems (NTUA)

Semester project for the **Robotics II: Intelligent Robotic Systems** course, School of Electrical & Computer Engineering, National Technical University of Athens (Spring 2026).

The project consists of two independent parts, both developed in **ROS 2**:

| Part | Topic | Simulator |
|------|-------|-----------|
| A | Mobile Robot – Autonomous Wall Following | Gazebo |
| B | Redundant Manipulator Control (Franka Emika Panda, 7-DOF) | RViz |

---

## Part A – Mobile Robot: Wall Following 🚗

Autonomous navigation of a **differential-drive mobile robot** that performs a full counter-clockwise (CCW) loop around a trapezoidal arrangement of walls, while maintaining a constant distance from them.

### Robot & Sensors
- Differential-drive platform (Gazebo model)
- **5× ultrasonic sonar sensors** (3 front, 2 lateral) for distance measurements
- **9-DOF IMU** (angular rates, linear accelerations, orientation)

### Control Architecture
The behavior is implemented as a **finite-state machine (FSM)**:

```
FIND_WALL  ─►  TURN  ─►  FOLLOW_WALL
                ▲             │
                └─────────────┘
```

- **`FIND_WALL`** – initial state: the robot drives until a wall is detected.
- **`TURN`** – transient state: rotates at the trapezoid corners.
- **`FOLLOW_WALL`** – main state, decomposed into:
  - *Primary task:* keep a constant lateral distance, staying parallel to the wall.
  - *Secondary task:* front-obstacle avoidance using the front sonars.

The initial yaw and the direction of travel (CCW) are parametrized by the team code:

```bash
ros2 launch mymobibot_gazebo mymobibot_world_wf.launch.py yaw_init:=1.292
```

### Evaluation
- Time-series analysis of lateral/front distances and linear/angular velocities
- Statistical error analysis of the distance-keeping performance
- Video of the full loop included in the report

---

## Part B – Redundant Manipulator: Task-Priority Control 🦾

Kinematic control of the **Franka Emika Panda** collaborative robot (7 revolute DOFs) executing **two simultaneous tasks** with strict priority, exploiting kinematic redundancy.

### Tasks
- **Primary task (m = 3):** periodic straight-line motion of the end-effector between two points `P_A = [0.617, −0.40, 0.199]` and `P_B = [0.617, +0.40, 0.199]` (position-only control → **4 redundant DOFs**).
- **Secondary task:** avoidance of **two static cylindrical obstacles**, resolved entirely in the **null space** of the primary task.

### Method Highlights
- **Trajectory generation:** three-phase motion profile (acceleration / cruise at `v_max = 0.1 m/s` / deceleration) using **quintic (5th-order) polynomials** for smooth position, velocity, and acceleration at the endpoints.
- **Differential kinematics:** geometric **Jacobian (3×7)** derived from the standard DH parameters of the Panda; computed algorithmically from the forward-kinematics transforms for real-time use.
- **Task-priority scheme:** secondary-task velocities are projected through the null-space projector `N = I − J⁺J`, so obstacle avoidance never disturbs the end-effector trajectory.
- **Obstacle avoidance:** **Artificial Potential Fields (APF)** – per-link distances to the obstacles generate repulsive velocities within an influence radius (0.15 m).
- **Approach phase:** a safe initialization strategy routes the end-effector through an intermediate waypoint (midpoint of the trajectory segment) before reaching `P_A`, preventing collisions from the home configuration.

### Evaluation
- Per-link minimum-distance plots identifying the critical links (elbow / Link 4)
- Joint position & velocity profiles, including discussion of **APF chattering** inside the influence zone
- Verification that the 7th joint remains idle (orientation not controlled), confirming an efficient solution

---

## Tech Stack

`ROS 2` · `Gazebo` · `RViz` · `Python` · `NumPy / Matplotlib`

## Repository Structure

```
.
├── part_a_wall_following/   # Mobile robot package (Gazebo)
├── part_b_manipulator/      # Panda task-priority controller (RViz)
├── reports/                 # Full technical reports (PDF, in Greek)
└── README.md
```

## Authors

- Panagiotis Georgiou
- Georgios Barkas
- Angelos Karavas

---

*Full theoretical analysis, derivations (DH parameters, Jacobian computation, trajectory equations), and simulation results are available in the PDF reports.*

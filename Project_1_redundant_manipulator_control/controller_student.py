#!/usr/bin/env python3
"""
=============================================================================
ROBOTICS II - NTUA
Redundant Manipulator Control - Trajectory following with Obstacle Avoidance
STUDENT TEMPLATE
=============================================================================

ASSIGNMENT OBJECTIVE:
Implement a kinematic controller for a 7-DOF robot manipulator that:
1. Tracks a linear trajectory between points PA and PB (PRIMARY TASK)
2. Avoids cylindrical obstacles using null-space control (SECONDARY TASK)

The end-effector performs periodic linear motion (position control only,
NOT orientation) while the redundant DOF is used for obstacle avoidance.

WHAT YOU NEED TO IMPLEMENT:
- compute_jacobian(): Geometric Jacobian matrix (3x7 for position)
- compute_primary_task_desired_velocity(): Desired end-effector velocity
- compute_secondary_task_desired_velocity(): Repulsive velocity in null-space
- control_loop(): Main control law combining primary and secondary tasks

PROVIDED FOR YOU:
- Forward kinematics (get_end_effector_position, get_link_positions)
- DH transformation matrix (dh_transform)
- Distance calculations (distance_point_to_cylinder, get_min_obstacle_distance)
- ROS2 infrastructure (publishers, visualization)
- Configuration loading from params.yaml

Author: Robotics II Course - NTUA
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point, Vector3, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from ament_index_python.packages import get_package_share_directory

import numpy as np
from numpy.linalg import pinv, norm, inv
from math import sin, cos, pi, sqrt
from datetime import datetime
from collections import deque
import yaml
import os
import math


# =============================================================================
#                               Franka Emika Panda
# =============================================================================
# Standard DH Convention: T = Rot_z(θ) * Trans_z(d) * Trans_x(a) * Rot_x(α)
# =============================================================================

STANDARD_DH_PARAMS = [
    # [a,       d,      alpha,    theta_offset]
    [0,       0.333,  -pi/2,    0],      # Joint 1
    [0,       0,       pi/2,    0],      # Joint 2
    [0.0825,  0.316,   pi/2,    0],      # Joint 3
    [-0.0825, 0,      -pi/2,    0],      # Joint 4
    [0,       0.384,   pi/2,    0],      # Joint 5
    [0.088,   0,       pi/2,    0],      # Joint 6
    [0,       0.107,   0,       0],      # Joint 7 (includes flange)
]

# =============================================================================
# URDF JOINT DEFINITIONS - for correct capsule frame transforms
# =============================================================================
# Each entry: (xyz translation, roll angle) — all joints have rpy=(roll,0,0)

# These are the joint origins defined in the Franka XACRO, which differ from the DH frames.
# The collision capsules are defined in these URDF joint frames.
# It uses modified DH parameters with joint frames at the joint origins.
# So the transforms are just the joint origin translations and roll rotations.
# The end-effector frame (link 7) is at the flange, which is 0.107 m along z from joint 7.
URDF_JOINTS = [
    ([0,    0,      0.333],  0),        # Joint 1
    ([0,    0,      0],     -pi/2),     # Joint 2
    ([0,   -0.316,  0],      pi/2),     # Joint 3
    ([0.0825, 0,    0],      pi/2),     # Joint 4
    ([-0.0825, 0.384, 0],  -pi/2),     # Joint 5
    ([0,    0,      0],      pi/2),     # Joint 6
    ([0.088, 0,     0],      pi/2),     # Joint 7
]

JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) #rads
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) #rads
CAPSULE_SAFETY_DISTANCE = 0.0

class CollisionCapsule:
    """
    Collision capsule (cylinder with hemispherical caps) from Franka XACRO.

    Each link has one or more capsules describing its actual collision geometry.
    """

    def __init__(self, xyz, radius, length, direction='z'):
        self.xyz = np.array(xyz)
        self.radius = radius
        self.length = length
        self.direction = direction

    def get_endpoints_in_link_frame(self):
        """Get the two endpoints of the capsule centerline in link frame."""
        half_len = self.length / 2.0
        if self.direction == 'x':
            offset = np.array([half_len, 0.0, 0.0])
        elif self.direction == 'y':
            offset = np.array([0.0, half_len, 0.0])
        else:  # 'z'
            offset = np.array([0.0, 0.0, half_len])
        return self.xyz - offset, self.xyz + offset


COLLISION_CAPSULES = {
    0: [CollisionCapsule([-0.075, 0, 0.06],    0.06  + CAPSULE_SAFETY_DISTANCE, 0.03,  'x')],
    1: [CollisionCapsule([0, 0, -0.1915],       0.06  + CAPSULE_SAFETY_DISTANCE, 0.283, 'z')],
    2: [CollisionCapsule([0, 0, 0],             0.06  + CAPSULE_SAFETY_DISTANCE, 0.12,  'z')],
    3: [CollisionCapsule([0, 0, -0.145],        0.06  + CAPSULE_SAFETY_DISTANCE, 0.15,  'z')],
    4: [CollisionCapsule([0, 0, 0],             0.06  + CAPSULE_SAFETY_DISTANCE, 0.12,  'z')],
    5: [CollisionCapsule([0, 0, -0.26],         0.06  + CAPSULE_SAFETY_DISTANCE, 0.10,  'z'),
        CollisionCapsule([0, 0.08, -0.13],      0.025 + CAPSULE_SAFETY_DISTANCE, 0.14,  'z')],
    6: [CollisionCapsule([0, 0, -0.03],         0.05  + CAPSULE_SAFETY_DISTANCE, 0.08,  'z')],
    7: [CollisionCapsule([0, 0, 0.01],          0.04  + CAPSULE_SAFETY_DISTANCE, 0.14,  'z'),
        CollisionCapsule([0.06, 0, 0.082],      0.03  + CAPSULE_SAFETY_DISTANCE, 0.01,  'x')],
}


class RedundantController(Node):
    """
    Kinematic controller for 7-DOF Panda manipulator with obstacle avoidance.
    
    Students implement the core control algorithms.
    """
    
    def __init__(self):
        super().__init__('redundant_controller')
        
        self.load_parameters()
        
        self.q = np.array(self.config['initial_joint_positions'])
        self.q_dot = np.zeros(7)
        
        self.trajectory_param = 0.0
        self.trajectory_direction = 1
        
        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)
        
        self.dt = 1.0 / self.config['control']['rate']
        self.timer = self.create_timer(self.dt, self.control_loop)

        # Set to False once you implement control_loop()
        self.bypass = False

        # Set to False until end-effector reaches PA to ensure proper start of timer
        self.trajectory_started = False

        # Set waypoints of initial path of end-effector from middle of trajectory to PA to hinder collision at start
        self.approach_waypoints = [(self.PA + self.PB) / 2.0, self.PA.copy()]

        # Index for which initial waypoint we are currently targeting
        self.approach_waypoint_idx = 0 

        self.log_dir = self.log_dir = os.path.expanduser('~/logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        max_log_entries = 100000  # ~16 min at 100 Hz
        # Using deque with maxlen to limit memory usage while still keeping recent history
        # You can log more or fewer variables as needed for analysis
        self.current_target = self.PA.copy()

        self.log = {
            'time': deque(maxlen=max_log_entries),
            'position': deque(maxlen=max_log_entries),
            'target_position': deque(maxlen=max_log_entries),
            'ee_velocity': deque(maxlen=max_log_entries),
            'q': deque(maxlen=max_log_entries),               
            'q_dot': deque(maxlen=max_log_entries),
            'min_obs_dist': deque(maxlen=max_log_entries),
            'obs1_dist': deque(maxlen=max_log_entries),
            'obs2_dist': deque(maxlen=max_log_entries),
            'manipulability': deque(maxlen=max_log_entries)
        }
        self.start_time = self.get_clock().now()
        
        self.get_logger().info('='*60)
        self.get_logger().info('Redundant Controller - STUDENT VERSION')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Trajectory: PA={self.PA} → PB={self.PB}')
        self.get_logger().info(f'Speed: {self.speed} m/s')
        self.get_logger().info(f'Obstacles: {len(self.obstacles)}')
        self.get_logger().info(f'Logs will be saved to: {self.log_dir}')
        self.get_logger().info('='*60)

    def load_parameters(self):
        try:
            pkg_share = get_package_share_directory('panda_redundant_controller_student')
            config_path = os.path.join(pkg_share, 'config', 'params.yaml')
            self.get_logger().info(f'Loading config from: {config_path}')
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                self.config = full_config['panda_controller']['ros__parameters']
        except Exception as e:
            self.get_logger().warn(f'Could not load config: {e}. Using defaults.')
            self.config = self._default_config()
        
        self.PA = np.array(self.config['trajectory']['PA'])
        self.PB = np.array(self.config['trajectory']['PB'])
        self.speed = self.config['trajectory']['speed']
        
        self.Kp = self.config['control']['Kp']
        self.Ko = self.config['control']['Ko']
        self.d_influence = self.config['control']['d_influence']
        self.damping = self.config['control']['damping']
        self.max_joint_vel = self.config['control']['max_joint_velocity']
        self.max_null_vel = self.config['control']['max_null_velocity']
        
        self.link_radius = self.config['robot']['link_radius']
        
        self.obstacles = []
        for pos in self.config['obstacles']['positions']:
            self.obstacles.append({
                'center': np.array(pos),
                'radius': self.config['obstacles']['radius'],
                'height': self.config['obstacles']['height']
            })
    
    def _default_config(self):
        return {
            'trajectory': {'PA': [0.617, -0.40, 0.199], 'PB': [0.617, 0.40, 0.199], 'speed': 0.1},
            'obstacles': {'radius': 0.05, 'height': 1.0,
                         'positions': [[0.30, -0.20, 0.50], [0.30, 0.20, 0.50]]},
            'robot': {'link_radius': 0.063},
            'control': {'rate': 100.0, 'Kp': 2.0, 'Ko': 0.3, 'd_influence': 0.15,
                       'damping': 0.01, 'max_joint_velocity': 0.5, 'max_null_velocity': 0.6},
            'initial_joint_positions': [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785],
            'visualization': {'gripper_opening': 0.04}
        }

    # =========================================================================
    # FORWARD KINEMATICS (PROVIDED)
    # =========================================================================
    
    def dh_transform(self, a, d, alpha, theta):
        """
        Compute homogeneous transformation matrix using Standard DH convention.
        
        T = Rot_z(θ) * Trans_z(d) * Trans_x(a) * Rot_x(α)
        
        Args:
            a: Link length (distance along x)
            d: Link offset (distance along z)
            alpha: Link twist (rotation about x)
            theta: Joint angle (rotation about z)
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        ca, sa = cos(alpha), sin(alpha)
        ct, st = cos(theta), sin(theta)
        
        return np.array([
            [ct,     -st*ca,   st*sa,   a*ct],
            [st,      ct*ca,  -ct*sa,   a*st],
            [0,       sa,      ca,      d   ],
            [0,       0,       0,       1   ]
        ])
    
    def get_all_transforms(self, q):
        """
        Compute transformation matrices from base to each frame.
        
        Args:
            q: Joint angles [7]
            
        Returns:
            List of 8 transformation matrices [T0, T1, T2, ..., T7]
            - T0 = Identity (base frame)
            - Ti = transform from base to frame i (i = 1..7)
            - T7 = end-effector frame
        """
        transforms = [np.eye(4)]  # T0 = base frame
        T = np.eye(4)
        
        for i in range(len(STANDARD_DH_PARAMS)):
            a, d, alpha, theta_offset = STANDARD_DH_PARAMS[i]
            theta = theta_offset + q[i]
            T = T @ self.dh_transform(a, d, alpha, theta)
            transforms.append(T.copy())
        
        return transforms
    
    def get_end_effector_position(self, q):
        """
        Compute end-effector (flange) position.
        
        Args:
            q: Joint angles [7]
            
        Returns:
            position: [x, y, z] in base frame
        """
        transforms = self.get_all_transforms(q)
        return transforms[-1][:3, 3]
    
    def get_link_positions(self, q):
        """
        Get positions of all link frames (for collision checking).
        
        Args:
            q: Joint angles [7]
            
        Returns:
            List of positions where pi is [x, y, z]
        """
        transforms = self.get_all_transforms(q)
        return [T[:3, 3] for T in transforms]

    def get_all_urdf_transforms(self, q):
        """
        Compute link frames using URDF joint definitions.

        URDF frames differ from DH frames in orientation. Capsule collision
        geometry is defined in URDF frames, so this method is needed for
        correct collision checking.

        Args:
            q: Joint angles [7]

        Returns:
            List of transformation matrices [T0, T1, ..., T7]
            - T0 = Identity (base frame)
            - Ti = URDF link i frame
        """
        transforms = [np.eye(4)]
        T = np.eye(4)
        for i in range(7):
            xyz, roll = URDF_JOINTS[i]
            cr, sr = cos(roll), sin(roll)
            T_origin = np.array([
                [1.0, 0.0, 0.0, xyz[0]],
                [0.0,  cr, -sr, xyz[1]],
                [0.0,  sr,  cr, xyz[2]],
                [0.0, 0.0, 0.0,   1.0],
            ])
            cq, sq = cos(q[i]), sin(q[i])
            T_joint = np.array([
                [cq, -sq, 0.0, 0.0],
                [sq,  cq, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
            T = T @ T_origin @ T_joint
            transforms.append(T.copy())
        return transforms

    # =========================================================================
    # DISTANCE CALCULATIONS (PROVIDED)
    # =========================================================================
    
    def distance_point_to_cylinder(self, point, cyl_center, cyl_radius, cyl_height):
        """
        Compute distance from a point to a vertical cylinder surface.
        
        The cylinder is oriented along the z-axis, centered at cyl_center.
        Handles three spatial regions: below, within, and above cylinder height.
        
        Args:
            point:      [x, y, z] query point (world frame)
            cyl_center: [x, y, z] center of cylinder (world frame)
            cyl_radius: radius of cylinder [m]
            cyl_height: total height of cylinder [m]
            
        Returns:
            distance: Signed distance to cylinder surface [m]
                      positive = point is OUTSIDE the cylinder
                      zero     = point is ON the surface
                      negative = point is INSIDE the cylinder (collision!)

        Geometry (3 cases):
            Below  (z < z_min): Euclidean distance to bottom rim edge
            Above  (z > z_max): Euclidean distance to top rim edge
            Inside (z_min <= z <= z_max): radial distance to curved surface
                                          = xy_dist - cyl_radius
        """
        z_min = cyl_center[2] - cyl_height / 2.0
        z_max = cyl_center[2] + cyl_height / 2.0
        
        # Radial distance in xy-plane
        xy_dist = norm(point[:2] - cyl_center[:2])
        
        if point[2] < z_min:
            # Below cylinder
            if xy_dist <= cyl_radius:
                return z_min - point[2]
            else:
                return sqrt((xy_dist - cyl_radius)**2 + (z_min - point[2])**2)
        elif point[2] > z_max:
            # Above cylinder
            if xy_dist <= cyl_radius:
                return point[2] - z_max
            else:
                return sqrt((xy_dist - cyl_radius)**2 + (point[2] - z_max)**2)
        else:
            # Within cylinder height range
            return xy_dist - cyl_radius
    
    def distance_segment_to_cylinder(self, p_start, p_end, cyl_center, cyl_radius, cyl_height, num_samples=5):
        """
        Compute minimum distance from a line segment to a vertical cylinder surface.

        Samples (num_samples + 2) points uniformly along the segment [p_start, p_end]
        and returns the minimum point-to-cylinder distance found, along with the
        parameter t of that closest point.

        Args:
            p_start:     [x, y, z] segment start point (world frame)
            p_end:       [x, y, z] segment end point (world frame)
            cyl_center:  [x, y, z] center of cylinder (world frame)
            cyl_radius:  radius of cylinder [m]
            cyl_height:  total height of cylinder [m]
            num_samples: number of intermediate sample points (default: 5)
                         total evaluated points = num_samples + 2 (includes endpoints)

        Returns:
            min_distance: minimum distance found along the segment to the cylinder surface [m]
                          positive = segment does not intersect cylinder
                          negative = segment penetrates cylinder (collision!)
            best_t:       parameter t in [0, 1] of the closest sampled point
                          t=0 → p_start, t=1 → p_end

        Note:
            Sampling approximation — true minimum may lie between sample points.
            Slightly overestimates distance near curved edges. Sufficient for
            velocity-level avoidance but not for exact collision detection.
        """
        min_dist = float('inf')
        best_t = 0.0
        for i in range(num_samples + 2):  # endpoints + intermediate
            t = i / (num_samples + 1)
            p = p_start + t * (p_end - p_start)
            d = self.distance_point_to_cylinder(p, cyl_center, cyl_radius, cyl_height)
            if d < min_dist:
                min_dist = d
                best_t = t
        return min_dist, best_t

    def get_min_obstacle_distance(self, q):
        """
        Find the minimum surface-to-surface distance between any robot capsule and any obstacle.

        Iterates over all collision capsules of all links, transforms them to world frame
        using URDF joint definitions, and computes the distance to each obstacle cylinder.
        The capsule radius is subtracted from the segment-to-cylinder distance to get the
        true surface-to-surface clearance.

        Args:
            q: Joint angles [7]

        Returns:
            min_distance:     Minimum surface-to-surface distance across all capsule-obstacle pairs [m]
                              positive = robot surface is outside all obstacles
                              zero     = robot surface is touching an obstacle
                              negative = COLLISION (surfaces overlap!)
            closest_link_idx: Index of the link (0-7) whose capsule is closest to an obstacle
                              -1 if no obstacles exist
            closest_obs_idx:  Index into self.obstacles of the nearest obstacle
                              -1 if no obstacles exist

        Note:
            Uses URDF transforms (get_all_urdf_transforms) — NOT DH transforms —
            because the collision capsule geometry is defined in URDF joint frames.
        """
        transforms = self.get_all_urdf_transforms(q)
        min_dist = float('inf')
        closest_link = -1
        closest_obs = -1

        for link_idx, capsules in COLLISION_CAPSULES.items():
            T = transforms[link_idx]
            R = T[:3, :3]
            t_vec = T[:3, 3]

            for capsule in capsules:
                p1_local, p2_local = capsule.get_endpoints_in_link_frame()
                p1_world = R @ p1_local + t_vec
                p2_world = R @ p2_local + t_vec

                for obs_idx, obs in enumerate(self.obstacles):
                    seg_dist, _ = self.distance_segment_to_cylinder(
                        p1_world, p2_world,
                        obs['center'], obs['radius'], obs['height']
                    )
                    dist = seg_dist - capsule.radius

                    if dist < min_dist:
                        min_dist = dist
                        closest_link = link_idx
                        closest_obs = obs_idx

        return min_dist, closest_link, closest_obs
    

    def get_closest_point_and_gradient(self, q):
        """
        Find the closest point on the robot to any obstacle and the escape direction.    
        Args:
            q: Joint angles [7]
            
        Returns:
            p_closest:   [x, y, z] position of closest point on robot (world frame)
            gradient:    [gx, gy, gz] unit vector pointing AWAY from obstacle
            distance:    Distance to obstacle (negative = collision!)
            link_idx:    Which link (0-7) the closest point is on
            safety_dist: capsule_radius + obstacle_radius
            
        Returns (None, None, inf, -1, 0.0) if no obstacles are within influence distance.
        """
        urdf_transforms = self.get_all_urdf_transforms(q)
        
        best_dist = float('inf')
        best_gradient = None
        best_link_idx = -1
        best_p_closest = None
        best_safety_dist = 0.0

        for link_idx, capsules in COLLISION_CAPSULES.items():
            T_urdf = urdf_transforms[link_idx]
            R = T_urdf[:3, :3]
            t_vec = T_urdf[:3, 3]

            for capsule in capsules:
                p1_local, p2_local = capsule.get_endpoints_in_link_frame()
                p1_world = R @ p1_local + t_vec
                p2_world = R @ p2_local + t_vec

                for obs in self.obstacles:
                    obs_center = obs['center']
                    cyl_radius = obs['radius']
                    cyl_height = obs['height']

                    seg_dist, seg_t = self.distance_segment_to_cylinder(
                        p1_world, p2_world, obs_center, cyl_radius, cyl_height
                    )
                    dist = seg_dist - capsule.radius

                    if dist < self.d_influence and dist > 0.001:
                        p_closest = p1_world + seg_t * (p2_world - p1_world)

                        # 3D repulsive gradient matching distance branches
                        z_min = obs_center[2] - cyl_height / 2.0
                        z_max = obs_center[2] + cyl_height / 2.0
                        xy_diff = p_closest[:2] - obs_center[:2]
                        xy_dist = norm(xy_diff)

                        if z_min <= p_closest[2] <= z_max:
                            if xy_dist > 0.001:
                                gradient = np.array([xy_diff[0]/xy_dist, xy_diff[1]/xy_dist, 0.0])
                            else:
                                gradient = np.array([1.0, 0.0, 0.0])
                        elif xy_dist <= cyl_radius:
                            gz = -1.0 if p_closest[2] < z_min else 1.0
                            gradient = np.array([0.0, 0.0, gz])
                        else:
                            gx = xy_diff[0] / xy_dist if xy_dist > 0.001 else 1.0
                            gy = xy_diff[1] / xy_dist if xy_dist > 0.001 else 0.0
                            gz = -1.0 if p_closest[2] < z_min else 1.0
                            gradient = np.array([gx, gy, gz])
                            g_norm = norm(gradient)
                            if g_norm > 0.001:
                                gradient /= g_norm

                        if dist < best_dist:
                            best_dist = dist
                            best_gradient = gradient
                            best_link_idx = link_idx
                            best_p_closest = p_closest
                            best_safety_dist = capsule.radius + cyl_radius

        return best_p_closest, best_gradient, best_dist, best_link_idx, best_safety_dist
    


    # =========================================================================
    # VISUALIZATION (PROVIDED)
    # =========================================================================
    
    def publish_joint_states(self):
        """Publish joint states for robot_state_publisher."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'fer_joint{i+1}' for i in range(7)] + \
                   ['fer_finger_joint1', 'fer_finger_joint2']
        finger_pos = self.config.get('visualization', {}).get('gripper_opening', 0.04)
        msg.position = self.q.tolist() + [finger_pos, finger_pos]
        msg.velocity = self.q_dot.tolist() + [0.0, 0.0]
        self.joint_pub.publish(msg)
    
    def publish_markers(self):
        """Publish visualization markers for trajectory, desired position, and collision capsules."""
        markers = MarkerArray()
        
        # Trajectory line (green)
        traj = Marker()
        traj.header.frame_id = "fer_link0"
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.ns = "trajectory"
        traj.id = 0
        traj.type = Marker.LINE_STRIP
        traj.action = Marker.ADD
        traj.scale.x = 0.01
        traj.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
        traj.points = [
            Point(x=self.PA[0], y=self.PA[1], z=self.PA[2]),
            Point(x=self.PB[0], y=self.PB[1], z=self.PB[2])
        ]
        markers.markers.append(traj)
        
        # Collision capsules (cyan semi-transparent cylinders)
        transforms = self.get_all_urdf_transforms(self.q)
        capsule_id = 0
        for link_idx, capsules in COLLISION_CAPSULES.items():
            T = transforms[link_idx]
            R = T[:3, :3]
            t_vec = T[:3, 3]

            for capsule in capsules:
                p1_local, p2_local = capsule.get_endpoints_in_link_frame()
                p1_world = R @ p1_local + t_vec
                p2_world = R @ p2_local + t_vec
                mid = (p1_world + p2_world) / 2.0
                seg = p2_world - p1_world
                seg_len = norm(seg)

                cap_m = Marker()
                cap_m.header.frame_id = "fer_link0"
                cap_m.header.stamp = self.get_clock().now().to_msg()
                cap_m.ns = "capsules"
                cap_m.id = capsule_id
                cap_m.type = Marker.CYLINDER
                cap_m.action = Marker.ADD
                cap_m.pose.position = Point(x=float(mid[0]), y=float(mid[1]), z=float(mid[2]))

                # Orient cylinder along the capsule segment direction
                if seg_len > 1e-6:
                    axis = seg / seg_len
                    # Cylinder default axis is Z; compute quaternion from Z to axis
                    z_axis = np.array([0.0, 0.0, 1.0])
                    cross = np.cross(z_axis, axis)
                    dot = np.dot(z_axis, axis)
                    if norm(cross) < 1e-6:
                        if dot > 0:
                            quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                        else:
                            quat = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
                    else:
                        w = 1.0 + dot
                        quat = Quaternion(x=float(cross[0]), y=float(cross[1]),
                                          z=float(cross[2]), w=float(w))
                        q_norm = sqrt(quat.x**2 + quat.y**2 + quat.z**2 + quat.w**2)
                        quat.x /= q_norm
                        quat.y /= q_norm
                        quat.z /= q_norm
                        quat.w /= q_norm
                else:
                    quat = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                cap_m.pose.orientation = quat
                cap_m.scale = Vector3(
                    x=float(capsule.radius * 2),
                    y=float(capsule.radius * 2),
                    z=float(seg_len + capsule.radius * 2)  # include hemispherical caps
                )
                cap_m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3)
                markers.markers.append(cap_m)
                capsule_id += 1

        self.marker_pub.publish(markers)
    
    def log_data(self):
        """Log data for analysis."""
        t = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        p = self.get_end_effector_position(self.q)
        J = self.compute_jacobian(self.q)
        
        # 1. Vecolity of end-effector
        v_ee = J @ self.q_dot
        
        # 2. W
        w = math.sqrt(max(0.0, np.linalg.det(J @ J.T))) 
        
        # 3. Distances of obstacles
        transforms = self.get_all_urdf_transforms(self.q)
        d1, d2 = float('inf'), float('inf')
        for link_idx, capsules in COLLISION_CAPSULES.items():
            T = transforms[link_idx]
            R, t_vec = T[:3, :3], T[:3, 3]
            for capsule in capsules:
                p1_local, p2_local = capsule.get_endpoints_in_link_frame()
                p1_world, p2_world = R @ p1_local + t_vec, R @ p2_local + t_vec
                

                sd1, _ = self.distance_segment_to_cylinder(
                    p1_world, p2_world, 
                    self.obstacles[0]['center'], self.obstacles[0]['radius'], self.obstacles[0]['height']
                )
                d1 = min(d1, sd1 - capsule.radius)
                
                sd2, _ = self.distance_segment_to_cylinder(
                    p1_world, p2_world, 
                    self.obstacles[1]['center'], self.obstacles[1]['radius'], self.obstacles[1]['height']
                )
                d2 = min(d2, sd2 - capsule.radius)
        
        min_dist = min(d1, d2)
        
        self.log['time'].append(t)
        self.log['position'].append(p.copy())
        self.log['target_position'].append(self.current_target.copy())
        self.log['ee_velocity'].append(v_ee.copy())
        self.log['q'].append(self.q.copy())              
        self.log['q_dot'].append(self.q_dot.copy())
        self.log['min_obs_dist'].append(min_dist)
        self.log['obs1_dist'].append(d1)
        self.log['obs2_dist'].append(d2)
        self.log['manipulability'].append(w)
    
    def save_log(self, filename=None):
        """Save logged data to file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'log_student.npz'
        filepath = os.path.join(self.log_dir, filename)
        np.savez(filepath, **{k: np.array(v) for k, v in self.log.items()})
        self.get_logger().info(f'Log saved to {filepath}')

    # =========================================================================
                            # TODO: IMPLEMENT THESE FUNCTIONS
    # =========================================================================
 
    def compute_jacobian(self, q):
        J = np.zeros((3, 7))
        transforms = self.get_all_transforms(q)
        p_ee = transforms[-1][:3, 3] # End-effector position (T7)

        for i in range(7):
            T_prev = transforms[i] # Frame of the previous joint
            b_prev = T_prev[:3, 2] # Rotation axis (Z)
            p_prev = T_prev[:3, 3] # Origin of the joint frame
            
            # Geometric Jacobian linear part: b_{i-1} x (p_ee - p_{i-1})
            J[:, i] = np.cross(b_prev, (p_ee - p_prev))
        return J
    
    def compute_approach_velocity(self):
        """
        Compute a velocity vector that moves the end-effector towards (PA+PB)/2 and PA.
        This is used at the start to ensure the robot does not collide with obstacle.
        """

        # If we have reached final waypoint (PA), return zero 
        if self.approach_waypoint_idx >= len(self.approach_waypoints):
          return np.zeros(3)
        
        # Get distance of End-Effector from current target waypoint
        target = self.approach_waypoints[self.approach_waypoint_idx]
        x_curr = self.get_end_effector_position(self.q)
        error = target - x_curr

        # If the distance is smaller than a threshold move to the next waypoint
        if norm(error) < 0.01:
            self.approach_waypoint_idx += 1

            # If both waypoints have been reached start main trajectory 
            if self.approach_waypoint_idx >= len(self.approach_waypoints):
                self.trajectory_started = True
                self.trajectory_param = 0.0
                return np.zeros(3)
            
            # If both waypoints have not been reached repeat for next waypoint
            target = self.approach_waypoints[self.approach_waypoint_idx]
            error = target - x_curr

        # P-Controller with speed saturation
        v = self.Kp * error
        v_norm = norm(v)
        if v_norm > self.speed:
            v = (v / v_norm) * self.speed
        
        self.current_target = target
        return v
     
    def compute_primary_task_desired_velocity(self):
        """
        Compute desired end-effector velocity for trajectory tracking.
        Implements a 5th-degree polynomial trajectory with 3 phases:
        Acceleration, Constant Velocity, and Deceleration.
        """
        # 1. Trajectory Parameters (As defined in Subchapter A.1)
        L = norm(self.PB - self.PA)  # Total path length
        v_max = self.speed           # Maximum constant velocity
        T_acc = 1.5                  # Acceleration/Deceleration duration (sec)
        s_acc = 0.1                  # Fraction of path (10%) covered during acceleration
        
        # 2. Total Time Calculation
        T_total = 2 * T_acc + (L - 2 * L * s_acc) / v_max
        
        # 3. Calculation of 5th Degree Polynomial Coefficients
        V_n = (v_max * T_acc) / L
        a3 = 10 * s_acc - 4 * V_n
        a4 = -15 * s_acc + 7 * V_n
        a5 = 6 * s_acc - 3 * V_n
        
        t = self.trajectory_param  # Current time elapsed
        
        # 4. Determine Phase (Acceleration, Constant Velocity, Deceleration)
        if t < T_acc:
            # PHASE 1: Acceleration
            tau = t / T_acc
            s = a3*tau**3 + a4*tau**4 + a5*tau**5
            # Derivative for velocity (applying chain rule: 1 / T_acc)
            s_dot_tau = 3*a3*tau**2 + 4*a4*tau**3 + 5*a5*tau**4
            s_dot = s_dot_tau / T_acc
            
        elif t < (T_total - T_acc):
            # PHASE 2: Constant Velocity
            s = s_acc + (v_max / L) * (t - T_acc)
            s_dot = v_max / L
            
        else:
            # PHASE 3: Deceleration (Using time reversal as derived in the report)
            tau_dec = (t - (T_total - T_acc)) / T_acc
            tau_inv = 1.0 - tau_dec
            
            P_acc_inv = a3*tau_inv**3 + a4*tau_inv**4 + a5*tau_inv**5
            s = 1.0 - P_acc_inv
            
            P_acc_inv_dot = 3*a3*tau_inv**2 + 4*a4*tau_inv**3 + 5*a5*tau_inv**4
            s_dot = P_acc_inv_dot / T_acc
            
        # 5. Compute Target Position and Feedforward Velocity
        x_target = self.PA + s * (self.PB - self.PA)
        v_ff = s_dot * (self.PB - self.PA)
        
        # 6. Closed-Loop Position Control (Feedback Term)
        x_curr = self.get_end_effector_position(self.q)
        v_fb = self.Kp * (x_target - x_curr)
        
        self.current_target = x_target
        return v_ff + v_fb
    
    def compute_secondary_task_desired_velocity(self, q):
        p_closest, grad, dist, link_idx, _ = self.get_closest_point_and_gradient(q)

        if p_closest is None or dist >= self.d_influence:
            return np.zeros(7)

        J_p = np.zeros((3, 7))
        transforms = self.get_all_transforms(q)
        
        active_joints = min(link_idx, 6)
        
        for i in range(active_joints + 1):
            T_i = transforms[i]
            z_i = T_i[:3, 2]
            p_i = T_i[:3, 3]
            J_p[:, i] = np.cross(z_i, (p_closest - p_i))

        d_safe    = max(dist, 0.005)
        amplitude = (self.Ko
                    * (1.0/d_safe - 1.0/self.d_influence)
                    * (1.0/d_safe**2))
        amplitude = min(amplitude, 50.0)
        
        F_rep   = amplitude * grad
        q_dot_0 = J_p.T @ F_rep
        
        return q_dot_0
    
    def control_loop(self):
        """
        Main control loop executed at a fixed frequency.
        Implements a Task-Priority Kinematic Strategy.
        """
        self.get_logger().info(f't={self.trajectory_param:.2f}')

        if self.bypass:
            self.publish_joint_states()
            self.publish_markers()
            self.log_data()
            return

        # 1. Compute kinematics and desired task velocities
        J = self.compute_jacobian(self.q)
        if not self.trajectory_started:
            x_dot_d = self.compute_approach_velocity()
        else:
            x_dot_d = self.compute_primary_task_desired_velocity()
        q_dot_0 = self.compute_secondary_task_desired_velocity(self.q)

        # 2. Compute Damped Least Squares (DLS) Pseudo-inverse
        # This prevents velocity spikes near singularities (Subchapter A.3)
        lambda_sq = self.damping**2
        J_pinv = J.T @ inv(J @ J.T + lambda_sq * np.eye(3))

        # 3. Primary Task: Particular Solution
        # This component ensures the end-effector follows the linear path
        q_dot_primary = J_pinv @ x_dot_d

        # 4. Secondary Task: Homogeneous Solution (Null-Space Projection)
        # We project the obstacle avoidance velocity into the null-space of the main task
        I = np.eye(7)
        N = I - J_pinv @ J                      # Null-space projector matrix
        q_dot_null = N @ q_dot_0                # Projected avoidance velocity

        # 5. Null-space Velocity Smoothing
        # Scale down null-space motion if it exceeds the user-defined safety limit
        null_norm = norm(q_dot_null)
        if null_norm > self.max_null_vel:
            q_dot_null = q_dot_null * (self.max_null_vel / null_norm)

        # 6. Final Control Law (Priority-based Summation)
        # Combined velocity: q_dot = q_dot_p + N * q_dot_0
        self.q_dot = q_dot_primary + q_dot_null
        
        # Apply safety saturation to all joint velocities
        self.q_dot = np.clip(self.q_dot, -self.max_joint_vel, self.max_joint_vel)
        
        # 7. Numerical Integration (Forward Euler)
        # Update joint positions: q(k+1) = q(k) + q_dot * dt
        self.q = self.q + self.q_dot * self.dt

        # 8. Trajectory Timer Management
        # Calculate total duration for a single pass from PA to PB
        L = norm(self.PB - self.PA)
        # Using the same formula as in Subchapter A.1
        T_total = 2 * 1.5 + (L - 2 * L * 0.1) / self.speed 

        # Increment trajectory parameter (time)
        # If End Effector has not reached PA yet, keep timer at zero 
        if self.trajectory_started:
            self.trajectory_param += self.dt
            
        # 9. Periodic Motion Logic
        # If the end of the segment is reached, reset timer and swap endpoints
        if self.trajectory_param >= T_total:
            self.trajectory_param = 0.0
            # Swap endpoints to enable returning motion
            self.PA, self.PB = self.PB.copy(), self.PA.copy()

        # Update ROS2 publishers and logging
        self.publish_joint_states()
        self.publish_markers()
        self.log_data()


def main(args=None):
    rclpy.init(args=args)
    controller = RedundantController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down...')
    finally:
        controller.save_log()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()








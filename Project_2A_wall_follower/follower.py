#!/usr/bin/env python3

"""
Start ROS node to publish linear and angular velocities to mymobibot in order to perform wall following.
"""

# ROS2 imports
import rclpy
from rclpy.node import Node

# Message types
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist

# Math imports
import numpy as np
import csv
import os
from datetime import datetime


def quaternion_to_euler(w, x, y, z):
    """
    Convert quaternion to Euler angles.
    Returns: (roll, pitch, yaw) in radians

    NOTE: This function is provided for you - no changes needed.
    """
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class WallFollower(Node):
    """
    Wall Following Robot using Finite State Machine.
    """

    STATE_FIND_WALL  = 0   # Drive forward
    STATE_TURN       = 1   # Rotate in place to become parallel to the wall
    STATE_FOLLOW_WALL = 2  # PID wall-following + front obstacle avoidance

    X = 3 + 9 + 5 # Our team number
    yaw_init = 1.292 # X modulo pi

    def __init__(self):
        super().__init__('wall_follower')

        # ========== PARAMETERS ==========
        self.declare_parameter('rate', 100.0)
        rate = self.get_parameter('rate').value

        # ========== SENSOR DATA ==========
        self.imu = Imu()           # IMU sensor message
        self.imu_yaw = 0.0         # Robot orientation (radians)
        self.sonar_F = 0.0         # Front sonar (distance in meters)
        self.sonar_FL = 0.0        # Front-left sonar (distance in meters)
        self.sonar_FR = 0.0        # Front-right sonar (distance in meters)
        self.sonar_L = 0.0         # Left sonar (distance in meters)
        self.sonar_R = 0.0         # Right sonar (distance in meters)
        self.sensors_ready = False
        
        # ========== CONTROL VARIABLES ==========
        self.velocity = Twist()
        self.period = 1.0 / rate
        
        # Time tracking
        self.time_prev = None
        self.initialized = False

        # ========== TEAM PARAMETERS =============

        # Given physical base inversion: a negative command turns CCW (Left).
        # Keeping -1 sets us up perfectly to follow the Right wall.
        self.rotation_dir = -1          # -1 = CCW (odd X), +1 = CW (even X)
 
        # Desired distance from the wall we are following (meters).
        self.desired_distance = 0.4
 
        # speeds
        self.base_speed = 0.3           # Forward speed (m/s)
        self.turn_speed = 0.35           # In-place rotation speed (rad/s)
 
        # Obstacle Thresholds
        self.front_wall_threshold = self.desired_distance - 0.2 # meters — triggers FIND_WALL → TURN
        self.front_safety_threshold = self.desired_distance + 0.2  # meters — activates front-avoidance blend in FOLLOW_WALL
 
        # When the front clears above this during TURN, we are parallel enough to start wall-following.
        self.front_clear_threshold = 1.0  # meters
 
        # PID CONTROLLER GAINS  (for wall-distance tracking)
        self.Kp = 1.8       # Proportional gain
        self.Ki = 0.25       # Integral gain 
        self.Kd = 0.3        # Derivative gain
        self.Ka = 0.6       # Wall-angle correction gain
 
        # FRONT-SONAR WEIGHTED AVOIDANCE GAIN
        self.K_avoid = 0.25
 
        # FSM STATE
        self.state = self.STATE_FIND_WALL
 
        # PID error state
        self.prev_error = 0.0
        self.integral_error = 0.0       # Integral accumulator
        self.integral_max = 1.5         # Anti-windup clamp
        self.filtered_d_error = 0.0     # Low-pass filtered derivative
        self.alpha_filter = 0.15        # Derivative filter smoothing (0=ignore new, 1=no filter)

        # Angle between side sonar and front-diagonal sonar (radians)
        # From the model, to calculate the real distance
        self.sensor_angle = np.radians(45)

        # Debug logging counter
        self.log_counter = 0
        self.log_interval = 10  # print every 10 cycles (= once per second at 100Hz)

        # ========== CSV LOGGING ==========
        log_dir = os.path.expanduser('~/ros2_ws/logs')
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'run_{datetime.now():%Y%m%d_%H%M%S}.csv')
        self.csv_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            't', 'state',
            'sonar_F', 'sonar_FL', 'sonar_FR', 'sonar_L', 'sonar_R',
            'imu_yaw',
            'side_dist', 'diag_dist', 'wall_dist', 'alpha',
            'error', 'P', 'I', 'D', 'A',
            'linear_x', 'angular_z'
        ])
        self.t_start = None
        self.get_logger().info(f'CSV log: {log_path}')

        # ========== ROS2 PUBLISHERS ==========
        self.velocity_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ========== ROS2 SUBSCRIBERS ==========
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.sonar_F_sub = self.create_subscription(LaserScan, '/sonar_F', self.sonar_front_callback, 10)
        self.sonar_FL_sub = self.create_subscription(LaserScan, '/sonar_FL', self.sonar_frontleft_callback, 10)
        self.sonar_FR_sub = self.create_subscription(LaserScan, '/sonar_FR', self.sonar_frontright_callback, 10)
        self.sonar_L_sub = self.create_subscription(LaserScan, '/sonar_L', self.sonar_left_callback, 10)
        self.sonar_R_sub = self.create_subscription(LaserScan, '/sonar_R', self.sonar_right_callback, 10)

        # ========== CONTROL TIMER ==========
        self.timer = self.create_timer(self.period, self.control_loop) 

        self.get_logger().info('=== Wall Follower Started ===')
        self.get_logger().info(f'Control rate: {rate}Hz')
        self.get_logger().info(f'Rotation direction: {"CCW" if self.rotation_dir == -1 else "CW"}')
        self.get_logger().info(f'Desired wall distance: {self.desired_distance} m')

    # ============ SENSOR CALLBACKS ============
    def imu_callback(self, msg):
        """Process IMU messages to get robot orientation."""
        (_, _, self.imu_yaw) = quaternion_to_euler(
            msg.orientation.w, msg.orientation.x,
            msg.orientation.y, msg.orientation.z)

    def sonar_front_callback(self, msg):
        """Process front sonar readings from LaserScan."""
        if len(msg.ranges) > 0:
            self.sonar_F = msg.ranges[2]
            self.check_sensors_ready()

    def sonar_frontleft_callback(self, msg):
        """Process front-left sonar readings from LaserScan."""
        if len(msg.ranges) > 0:
            self.sonar_FL = msg.ranges[2]
            self.check_sensors_ready()

    def sonar_frontright_callback(self, msg):
        """Process front-right sonar readings from LaserScan."""
        if len(msg.ranges) > 0:
            self.sonar_FR = msg.ranges[2]
            self.check_sensors_ready()

    def sonar_left_callback(self, msg):
        """Process left sonar readings from LaserScan."""
        if len(msg.ranges) > 0:
            self.sonar_L = msg.ranges[2]
            self.check_sensors_ready()

    def sonar_right_callback(self, msg):
        """Process right sonar readings from LaserScan."""
        if len(msg.ranges) > 0:
            self.sonar_R = msg.ranges[2]
            self.check_sensors_ready()

    def check_sensors_ready(self):
        """Check if all sensors have received data."""
        if all(val > 0 for val in [self.sonar_F, self.sonar_FL, self.sonar_FR, self.sonar_L, self.sonar_R]):
            self.sensors_ready = True

    # =========== HELPER: side sonar for wall-following =========
    def get_wall_side_sonar(self):
        """
        Return the sonar reading from the side we are following.
        """
        if self.rotation_dir == +1:  # CW, our wall on left
            return self.sonar_L
        else:                        # CCW, our wall on right
            return self.sonar_R
        
    # =========== HELPER: front-diagonal sonar on wall side =========
    def get_wall_front_diagonal_sonar(self):
        """
        Return the front-diagonal sonar on the wall side.
        """
        if self.rotation_dir == +1:  # CW, our wall on left
            return self.sonar_FL
        else:                        # CCW, our wall on right
            return self.sonar_FR    

    # =========== HELPER: state name for logging =========
    def state_name(self):
 
        names = {0: "FIND_WALL", 1: "TURN", 2: "FOLLOW_WALL"}
        return names.get(self.state, "UNKNOWN")
        
    # ============ CONTROL LOOP ============
    def control_loop(self):
        """
        Main control loop implementing the wall-following behavior
        """
        # Wait for all sensors to be ready
        if not self.sensors_ready:
            return
        
        # Initialization on first run
        if not self.initialized:
            self.velocity.linear.x = 0.0
            self.velocity.angular.z = 0.0
            self.time_prev = self.get_clock().now()
            self.initialized = True
            self.get_logger().info("All sensors ready! Starting wall follower...")
            return
        
        # Calculate time interval
        time_now = self.get_clock().now()
        dt = (time_now - self.time_prev).nanoseconds / 1e9
        self.time_prev = time_now

        # Safety: skip if dt is zero or unreasonably large
        if dt <= 0.0 or dt > 1.0:
            return
        
        # Capture start time for CSV
        if self.t_start is None:
            self.t_start = self.get_clock().now()
        
        # Extract current sonar readings
        sonar_front      = self.sonar_F
        sonar_frontleft  = self.sonar_FL
        sonar_frontright = self.sonar_FR
        sonar_left       = self.sonar_L
        sonar_right      = self.sonar_R

        # Identify which diagonal points to the open space vs the wall
        if self.rotation_dir == -1:  # CCW, Right wall
            open_diag = sonar_frontleft
        else:                        # CW, Left wall
            open_diag = sonar_frontright

        # For front obstacle avoidance, ONLY look at the front and the open side.
        # Ignore the wall-side diagonal, otherwise the wall itself triggers avoidance!
        if self.state == self.STATE_FOLLOW_WALL:
            front_obstacle_dist = min(sonar_front, open_diag)
        else:
            # When finding the wall, check all three
            front_obstacle_dist = min(sonar_front, sonar_frontleft, sonar_frontright)

        # -----------------------------------------------------------------
        # PERIODIC DEBUG LOG
        # -----------------------------------------------------------------
        follow_ran = False
        self.log_counter += 1
        if self.log_counter >= self.log_interval:
            self.log_counter = 0
            
            # Print physical state and motor output
            self.get_logger().info(
                f'[{self.state_name()}] '
                f'Front={front_obstacle_dist:.2f}m, Side={self.get_wall_side_sonar():.2f}m | '
                f'Speed: {self.velocity.linear.x:.2f} m/s, Steer: {self.velocity.angular.z:+.2f} rad/s'
            )

        # ====== THE STATE MACHINE LOGIC ============

        if self.state == self.STATE_FIND_WALL:

            self.velocity.linear.x = self.base_speed
            self.velocity.angular.z = 0.0
 
            # Transition: to make a turn, we are at a corner, or first detect the wall in front.
            if front_obstacle_dist < self.front_wall_threshold:
                self.state = self.STATE_TURN
                self.prev_error = 0.0
                self.integral_error = 0.0
                self.filtered_d_error = 0.0
                self.get_logger().info(f'FIND_WALL ==> TURN  (front_obstacle_dist={front_obstacle_dist:.2f} m)')
        
        elif self.state == self.STATE_TURN:

            self.velocity.linear.x = 0.0
            self.velocity.angular.z = self.rotation_dir * self.turn_speed
 
            side_dist = self.get_wall_side_sonar()
            diag_dist = self.get_wall_front_diagonal_sonar()

            alpha = np.arctan2(
                - diag_dist * np.cos(self.sensor_angle) + side_dist,
                diag_dist * np.sin(self.sensor_angle))
            
            # exit: front clear, big distance, AND side sonar sees the wall nearby

            # if (sonar_front > self.front_clear_threshold and side_dist < self.desired_distance + 0.4):
            if (sonar_front > self.front_clear_threshold
            and side_dist < self.desired_distance + 0.4
            and abs(alpha) < 0.2): # roughly parallel
                self.state = self.STATE_FOLLOW_WALL
                wall_dist_est = side_dist * np.cos(alpha)
                self.prev_error = self.desired_distance - wall_dist_est
                self.integral_error = 0.0
                self.filtered_d_error = 0.0
                self.post_turn_counter = 50
                self.get_logger().info(f'(front={sonar_front:.2f}, side={side_dist:.2f})')

            if self.log_counter == 0:
                self.get_logger().info(
                    f'[TURN] front={sonar_front:.2f}m (need > {self.front_clear_threshold:.1f}) | '
                    f'side={side_dist:.2f}m (need < {self.desired_distance + 0.4:.1f}) | '
                    f'alpha={alpha:+.2f} rad (need < 0.25)'
                )
            
            follow_ran = False
        elif self.state == self.STATE_FOLLOW_WALL:
            follow_ran = True
            # --- Check for hard corner: front very close ==> TURN, the FSM changing from TURN ==> WALL FOLLOWING ==> TURN ==> ...
            if front_obstacle_dist < self.front_wall_threshold * 1.05:
                self.state = self.STATE_TURN
                self.prev_error = 0.0
                self.integral_error = 0.0
                self.filtered_d_error = 0.0
                self.get_logger().info(f'FOLLOW_WALL ==> TURN  (front_obstacle_dist={front_obstacle_dist:.2f} m, hard corner)')
                self.velocity.linear.x = 0.0
                self.velocity.angular.z = 0.0
                # Log this cycle before returning
                t_now = (self.get_clock().now() - self.t_start).nanoseconds / 1e9
                self.csv_writer.writerow([
                    f'{t_now:.3f}', 'TURN',
                    f'{sonar_front:.3f}', f'{sonar_frontleft:.3f}', f'{sonar_frontright:.3f}',
                    f'{sonar_left:.3f}', f'{sonar_right:.3f}', f'{self.imu_yaw:.4f}',
                    f'{self.get_wall_side_sonar():.3f}', f'{self.get_wall_front_diagonal_sonar():.3f}',
                    f'{self.get_wall_side_sonar():.3f}', '0.0000',
                    '0.0000', '0.0000', '0.0000', '0.0000', '0.0000',
                    '0.000', '0.000'
                ])
                self.velocity_pub.publish(self.velocity)
                return
 
            # =========================================================
            #  PRIMARY Task : PID wall-distance controller

            side_dist = self.get_wall_side_sonar()
            diag_dist = self.get_wall_front_diagonal_sonar()

            # Estimate robot's angle relative to the wall using two sensors
            # alpha > 0 means that the robot is angled toward the wall and alpha < 0 → robot is angled away from the wall
            
            alpha = np.arctan2(
                - diag_dist * np.cos(self.sensor_angle) + side_dist,
                diag_dist * np.sin(self.sensor_angle))

            # Project the true perpendicular wall distance
            wall_dist = side_dist * np.cos(alpha)

            # error for pid, if positive, we are too close
            error = self.desired_distance - wall_dist

            # Integral of error, clamped to prevent windup
            self.integral_error += error * dt
            self.integral_error = np.clip(self.integral_error, -self.integral_max, self.integral_max)

            # Filtered derivative of error (low-pass to reduce noise)
            # Technique to avoid derivative spikes, Aggele thn exo xanaxrisimopoiisi googlare ti
            raw_d_error = (error - self.prev_error) / dt
            self.filtered_d_error = (self.alpha_filter * raw_d_error + (1 - self.alpha_filter) * self.filtered_d_error)

            self.prev_error = error

            # PID output + wall-angle correction, multiplied with my direction again

            angular_pid = self.rotation_dir * (
                self.Kp * error +
                self.Ki * self.integral_error +
                self.Kd * self.filtered_d_error +
                self.Ka * alpha)

            # =========================================================
            #  SECONDARY Task: front-sonar weighted avoidance
            # Opos einai tora, den kanei kai polla to secondary task, kanei ligo blend to transition.
            # Kata vasi allazei state to FSM, eidika stis 2 apotomes gonies

            if front_obstacle_dist < self.front_safety_threshold:

                inv_FL = 1.0 / max(sonar_frontleft, 0.01)
                inv_F  = 1.0 / max(sonar_front, 0.01)
                inv_FR = 1.0 / max(sonar_frontright, 0.01)

                # Steering signal: asymmetry between left and right
                turn_signal = inv_FL - inv_FR

                # If FR is closer -> turn_signal is negative -> angular_avoid is negative -> turns Left safely.
                angular_avoid = self.K_avoid * turn_signal
                
                # Center wall naturally pushes us Left (CCW) into our wall-following loop
                angular_center = self.rotation_dir * self.K_avoid * inv_F * 0.5

                # Slow down proportionally to front distance
                speed_factor = front_obstacle_dist / self.front_safety_threshold
                linear_x = max(0.1, self.base_speed * speed_factor)

                # Combine PID + avoidance
                angular_z = angular_pid + angular_avoid + angular_center

            else:

                # Slow down when distance error is large to give controller time
                linear_x = self.base_speed * max(0.6, 1.0 - 1.5 * abs(error))
                angular_z = angular_pid

            # --- Clamp velocities to safe ranges ---
            angular_z = np.clip(angular_z, -0.8, 0.8)
 
            self.velocity.linear.x = float(linear_x)
            self.velocity.angular.z = float(angular_z)

            # --- PERIODIC CONTROL LOG ---
            if self.log_counter == 0:
                self.get_logger().info(
                    f'[FOLLOW] err={error:+.2f}m | '
                    f'P={self.Kp * error:+.2f} '
                    f'I={self.Ki * self.integral_error:+.2f} '
                    f'D={self.Kd * self.filtered_d_error:+.2f} '
                    f'A={self.Ka * alpha:+.2f} | '
                    f'cmd_z={angular_z:+.2f} rad/s'
                )
                
        # ========== CSV ROW ==========
        t_now = (self.get_clock().now() - self.t_start).nanoseconds / 1e9
        if follow_ran:
            log_side  = side_dist
            log_diag  = diag_dist
            log_wall  = wall_dist
            log_alpha = alpha
            log_err   = error
            log_P     = self.Kp * error
            log_I     = self.Ki * self.integral_error
            log_D     = self.Kd * self.filtered_d_error
            log_A     = self.Ka * alpha
        else:
            log_side  = self.get_wall_side_sonar()
            log_diag  = self.get_wall_front_diagonal_sonar()
            log_wall  = log_side
            log_alpha = 0.0
            log_err   = 0.0
            log_P = log_I = log_D = log_A = 0.0
 
        self.csv_writer.writerow([
            f'{t_now:.3f}', self.state_name(),
            f'{sonar_front:.3f}', f'{sonar_frontleft:.3f}', f'{sonar_frontright:.3f}',
            f'{sonar_left:.3f}', f'{sonar_right:.3f}',
            f'{self.imu_yaw:.4f}',
            f'{log_side:.3f}', f'{log_diag:.3f}', f'{log_wall:.3f}', f'{log_alpha:.4f}',
            f'{log_err:.4f}', f'{log_P:.4f}', f'{log_I:.4f}', f'{log_D:.4f}', f'{log_A:.4f}',
            f'{self.velocity.linear.x:.3f}', f'{self.velocity.angular.z:.3f}'
        ])    

        # Publish velocity command
        self.velocity_pub.publish(self.velocity)


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)
    follower = WallFollower()
 
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        follower.get_logger().info('Shutting down...')
    finally:
        try:
            follower.csv_file.close()
            follower.get_logger().info('CSV log saved.')
        except Exception:
            pass
        follower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
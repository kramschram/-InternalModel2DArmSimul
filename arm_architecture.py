"""Control nodes and biomechanics for a delayed two-joint, four-muscle arm."""
from collections import deque
from dataclasses import dataclass
import numpy as np


class DelayLine:
    def __init__(self, delay_s, dt, initial):
        n = max(0, round(delay_s / dt)); x = np.asarray(initial, float)
        self.queue = deque([x.copy() for _ in range(n + 1)], maxlen=n + 1)

    def push(self, value):
        self.queue.append(np.asarray(value, float).copy())
        return self.queue[0].copy()


@dataclass
class ArmState:
    q: np.ndarray
    dq: np.ndarray
    activation: np.ndarray

    def copy(self):
        return ArmState(self.q.copy(), self.dq.copy(), self.activation.copy())


class DesiredStateNode:
    def __init__(self, wrist_xy):
        self.wrist_xy = np.asarray(wrist_xy, float)


class TwoJointFourMuscleArm:
    """Shoulder/elbow arm actuated by flexor-extensor pairs at each joint."""
    def __init__(self, c):
        self.l1, self.l2 = c["link_lengths_m"]
        self.inertia = np.asarray(c["joint_inertia_kg_m2"], float)
        self.damping = np.asarray(c["joint_damping_nms_rad"], float)
        self.stiffness = np.asarray(c["passive_stiffness_nm_rad"], float)
        self.rest = np.asarray(c["rest_angles_rad"], float)
        self.limits = np.asarray(c["joint_limits_rad"], float)
        self.max_torque = np.asarray(c["maximum_muscle_torque_nm"], float)
        self.tau_on = c["activation_time_constant_s"]
        self.tau_off = c["deactivation_time_constant_s"]
        self.initial = ArmState(np.asarray(c["initial_angles_rad"], float),
                                np.asarray(c["initial_velocity_rad_s"], float), np.zeros(4))

    def wrist(self, q):
        q1, q2 = q
        return np.array([self.l1*np.cos(q1)+self.l2*np.cos(q1+q2),
                         self.l1*np.sin(q1)+self.l2*np.sin(q1+q2)])

    def points(self, q):
        elbow = self.l1*np.array([np.cos(q[0]), np.sin(q[0])])
        return np.zeros(2), elbow, self.wrist(q)

    def inverse_kinematics(self, xy, elbow_solution="down"):
        x, y = xy; radius = np.hypot(x, y)
        if not abs(self.l1-self.l2) <= radius <= self.l1+self.l2:
            raise ValueError(f"Target radius {radius:.3f} m is unreachable")
        c2 = np.clip((x*x+y*y-self.l1**2-self.l2**2)/(2*self.l1*self.l2), -1, 1)
        q2 = np.arccos(c2) * (-1 if elbow_solution == "up" else 1)
        q1 = np.arctan2(y, x)-np.arctan2(self.l2*np.sin(q2), self.l1+self.l2*np.cos(q2))
        return np.array([q1, q2])

    def muscle_torques(self, activation):
        """Signed torque contribution of each muscle: SF, SE, EF, EE."""
        signs = np.array([1.0, -1.0, 1.0, -1.0])
        return signs * self.max_torque * np.asarray(activation)

    def joint_torques(self, activation):
        mt = self.muscle_torques(activation)
        return np.array([mt[0]+mt[1], mt[2]+mt[3]])

    def step(self, state, excitation, dt):
        u = np.clip(excitation, 0, 1)
        tau = np.where(u > state.activation, self.tau_on, self.tau_off)
        activation = np.clip(state.activation + dt*(u-state.activation)/tau, 0, 1)
        torque = self.joint_torques(activation)
        ddq = (torque-self.damping*state.dq-self.stiffness*(state.q-self.rest))/self.inertia
        dq = state.dq + dt*ddq; q = state.q + dt*dq
        for j in range(2):
            lo, hi = self.limits[j]
            if q[j] < lo: q[j], dq[j] = lo, max(0, dq[j])
            elif q[j] > hi: q[j], dq[j] = hi, min(0, dq[j])
        return ArmState(q, dq, activation)


class InverseModel:
    def __init__(self, arm, c):
        self.arm = arm; self.kp = np.asarray(c["kp_nm_rad"], float)
        self.kd = np.asarray(c["kd_nms_rad"], float); self.cocontraction = c["baseline_cocontraction"]

    def desired_joints(self, wrist_xy, elbow_solution):
        return self.arm.inverse_kinematics(wrist_xy, elbow_solution)

    def motor_command(self, desired_q, estimated_q, estimated_dq):
        tau = self.kp*(desired_q-estimated_q)-self.kd*estimated_dq
        u = np.full(4, self.cocontraction)
        u[0] += max(tau[0], 0)/self.arm.max_torque[0]; u[1] += max(-tau[0], 0)/self.arm.max_torque[1]
        u[2] += max(tau[1], 0)/self.arm.max_torque[2]; u[3] += max(-tau[1], 0)/self.arm.max_torque[3]
        return np.clip(u, 0, 1)


class SensoryFeedbackNode:
    """Receives delayed joint angles and reconstructs wrist position."""
    def __init__(self, arm, dt, velocity_smoothing):
        self.arm, self.dt, self.s = arm, dt, velocity_smoothing
        self.previous = None; self.filtered_dq = np.zeros(2)

    def decode(self, delayed_q):
        q = np.asarray(delayed_q, float)
        raw = np.zeros(2) if self.previous is None else (q-self.previous)/self.dt
        self.filtered_dq = self.s*self.filtered_dq + (1-self.s)*raw
        self.previous = q.copy()
        return q, self.filtered_dq.copy(), self.arm.wrist(q)


class ForwardModel:
    def __init__(self, arm, initial_state, correction_gain):
        self.arm, self.state, self.gain = arm, initial_state.copy(), correction_gain

    def update(self, efference_copy, sensed_q, sensed_dq, dt):
        self.state = self.arm.step(self.state, efference_copy, dt)
        self.state.q += self.gain*(sensed_q-self.state.q)
        self.state.dq += self.gain*(sensed_dq-self.state.dq)
        return self.state.copy(), self.arm.wrist(self.state.q)


class GoalComparator:
    def __init__(self, tolerance_m, hold_time_s, dt):
        self.tolerance = tolerance_m; self.required = max(1, round(hold_time_s/dt)); self.count = 0

    def update(self, desired_xy, actual_feedback_xy):
        error = float(np.linalg.norm(desired_xy-actual_feedback_xy))
        self.count = self.count + 1 if error <= self.tolerance else 0
        return error, self.count >= self.required

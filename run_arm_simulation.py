"""Animate arm, target error, joint torques, and four muscle activations together."""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from arm_architecture import (DelayLine, DesiredStateNode, ForwardModel, GoalComparator,
                              InverseModel, SensoryFeedbackNode, TwoJointFourMuscleArm)


def load_config(path):
    with open(path, encoding="utf-8") as f: return json.load(f)


def simulate(c, internal_override=None):
    dt = c["simulation"]["dt_s"]; n = round(c["simulation"]["duration_s"]/dt)+1
    rng = np.random.default_rng(c["simulation"]["random_seed"])
    arm = TwoJointFourMuscleArm(c["arm"]); state = arm.initial.copy()
    desired = DesiredStateNode([c["target"]["x_m"], c["target"]["y_m"]])
    inverse = InverseModel(arm, c["inverse_model"])
    q_des = inverse.desired_joints(desired.wrist_xy, c["target"]["elbow_solution"])
    sensory = SensoryFeedbackNode(arm, dt, c["sensory_feedback"]["velocity_smoothing"])
    comparator = GoalComparator(c["goal"]["tolerance_m"], c["goal"]["hold_time_s"], dt)
    use_internal = c["internal_model"]["enabled"] if internal_override is None else internal_override
    forward = ForwardModel(arm, state, c["internal_model"]["sensory_correction_gain"])
    motor_delay = DelayLine(c["delays"]["motor_s"], dt, np.zeros(4))
    sensory_delay = DelayLine(c["delays"]["sensory_s"], dt, state.q)
    motor_sd = np.asarray(c["noise"]["motor_command_sd"]); angle_sd = np.asarray(c["noise"]["joint_angle_feedback_sd_rad"])

    names = ["q","wrist","wrist_feedback","wrist_pred","activation","joint_torque","muscle_torque","command"]
    r = {k: [] for k in names}; r.update(t=[], error=[], achieved=[])
    sensed_q, sensed_dq = state.q.copy(), state.dq.copy(); est_q, est_dq = sensed_q.copy(), sensed_dq.copy()
    for i in range(n):
        clean = inverse.motor_command(q_des, est_q, est_dq)
        noisy = clean + (rng.normal(0, motor_sd, 4) if c["noise"]["enabled"] else 0)
        state = arm.step(state, motor_delay.push(np.clip(noisy, 0, 1)), dt)
        feedback_q = state.q + (rng.normal(0, angle_sd, 2) if c["noise"]["enabled"] else 0)
        sensed_q, sensed_dq, sensed_xy = sensory.decode(sensory_delay.push(feedback_q))
        if use_internal:
            predicted, predicted_xy = forward.update(clean, sensed_q, sensed_dq, dt)
            est_q, est_dq = predicted.q, predicted.dq
        else:
            predicted_xy = sensed_xy.copy(); est_q, est_dq = sensed_q, sensed_dq
        error, achieved = comparator.update(desired.wrist_xy, sensed_xy)
        r["t"].append(i*dt); r["q"].append(state.q.copy()); r["wrist"].append(arm.wrist(state.q))
        r["wrist_feedback"].append(sensed_xy); r["wrist_pred"].append(predicted_xy)
        r["activation"].append(state.activation.copy()); r["joint_torque"].append(arm.joint_torques(state.activation))
        r["muscle_torque"].append(arm.muscle_torques(state.activation)); r["command"].append(clean)
        r["error"].append(error); r["achieved"].append(achieved)
    for k in r: r[k] = np.asarray(r[k])
    r.update(arm=arm, target=desired.wrist_xy, q_des=q_des, internal=use_internal)
    return r


def animate(r, c, save=None):
    arm, t = r["arm"], r["t"]
    stride = max(1, round(c["animation"]["interval_s"]/c["simulation"]["dt_s"])); frames = np.arange(0, len(t), stride)
    fig = plt.figure(figsize=(13, 8)); gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1])
    ax_arm = fig.add_subplot(gs[:,0]); ax_err = fig.add_subplot(gs[0,1]); ax_tau = fig.add_subplot(gs[0,2])
    ax_act = fig.add_subplot(gs[1,1:])
    reach = arm.l1+arm.l2
    ax_arm.set(xlim=(-1.1*reach,1.1*reach),ylim=(-1.1*reach,1.1*reach),aspect="equal",xlabel="x (m)",ylabel="y (m)",title="Two-joint arm")
    ax_arm.grid(alpha=.25); ax_arm.plot(*r["target"],"r*",ms=16,label="Desired wrist")
    body, = ax_arm.plot([],[],"o-",lw=6,label="Arm"); feedback, = ax_arm.plot([],[],"s",label="Actual feedback")
    prediction, = ax_arm.plot([],[],"^",label="Forward prediction"); trail, = ax_arm.plot([],[],alpha=.3)
    note = ax_arm.text(.02,.98,"",transform=ax_arm.transAxes,va="top"); ax_arm.legend(loc="lower left",fontsize=8)

    ax_err.set(xlim=(0,t[-1]),ylim=(0,max(.02,1.05*r["error"].max())),title="Error to target",xlabel="time (s)",ylabel="error (m)"); ax_err.grid(alpha=.25)
    error_line, = ax_err.plot([],[],color="tab:red"); ax_err.axhline(c["goal"]["tolerance_m"],ls="--",color="gray",label="tolerance"); ax_err.legend(fontsize=8)

    torque_limit = max(1.0, 1.1*np.abs(r["joint_torque"]).max())
    ax_tau.set(xlim=(0,t[-1]),ylim=(-torque_limit,torque_limit),title="Net joint torques",xlabel="time (s)",ylabel="torque (N m)"); ax_tau.grid(alpha=.25); ax_tau.axhline(0,color="gray",lw=.8)
    shoulder_tau, = ax_tau.plot([],[],label="Shoulder"); elbow_tau, = ax_tau.plot([],[],label="Elbow"); ax_tau.legend(fontsize=8)

    labels = ["Shoulder flexor","Shoulder extensor","Elbow flexor","Elbow extensor"]
    ax_act.set(xlim=(0,t[-1]),ylim=(0,1.05),title="Muscle activations",xlabel="time (s)",ylabel="activation"); ax_act.grid(alpha=.25)
    activation_lines = [ax_act.plot([],[],label=label)[0] for label in labels]; ax_act.legend(fontsize=8,ncol=4)

    def update(frame):
        i = int(frames[frame]); s,e,w = arm.points(r["q"][i]); body.set_data([s[0],e[0],w[0]],[s[1],e[1],w[1]])
        feedback.set_data([r["wrist_feedback"][i,0]],[r["wrist_feedback"][i,1]])
        prediction.set_data([r["wrist_pred"][i,0]],[r["wrist_pred"][i,1]]); prediction.set_visible(r["internal"])
        trail.set_data(r["wrist"][:i+1,0],r["wrist"][:i+1,1]); error_line.set_data(t[:i+1],r["error"][:i+1])
        shoulder_tau.set_data(t[:i+1],r["joint_torque"][:i+1,0]); elbow_tau.set_data(t[:i+1],r["joint_torque"][:i+1,1])
        for j,line in enumerate(activation_lines): line.set_data(t[:i+1],r["activation"][:i+1,j])
        note.set_text(f"t={t[i]:.2f} s\ninternal model={'ON' if r['internal'] else 'OFF'}\ngoal={'ACHIEVED' if r['achieved'][i] else 'not yet'}")
        return body,feedback,prediction,trail,error_line,shoulder_tau,elbow_tau,note,*activation_lines

    ani = FuncAnimation(fig,update,frames=len(frames),interval=1000*c["animation"]["interval_s"],blit=False)
    fig.tight_layout()
    if save:
        if Path(save).suffix.lower() != ".gif": raise ValueError("Use a .gif output filename")
        ani.save(save,writer=PillowWriter(fps=round(1/c["animation"]["interval_s"])),dpi=c["animation"]["dpi"])
    else: plt.show()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="architecture_config.json"); p.add_argument("--save"); p.add_argument("--no-animation",action="store_true")
    g=p.add_mutually_exclusive_group(); g.add_argument("--internal-model",action="store_true"); g.add_argument("--no-internal-model",action="store_true"); a=p.parse_args()
    override=True if a.internal_model else False if a.no_internal_model else None
    c=load_config(a.config); r=simulate(c,override); reached=np.flatnonzero(r["achieved"])
    print("internal model:",r["internal"]); print("final feedback error (cm):",100*r["error"][-1]); print("goal achieved:",bool(reached.size))
    if reached.size: print("first sustained achievement (s):",r["t"][reached[0]])
    if not a.no_animation: animate(r,c,a.save)
if __name__ == "__main__": main()

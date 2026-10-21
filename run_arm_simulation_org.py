"""Simulate and animate the architecture in InternalModel.pdf."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from arm_architecture import (DelayLine, DesiredStateNode, ForwardModel, GoalComparator,
                              InverseModel, SensoryFeedbackNode, TwoJointFourMuscleArm)


def load(path):
    with open(path, encoding="utf-8") as f: return json.load(f)


def simulate(c, internal_override=None):
    dt = c["simulation"]["dt_s"]; n = int(c["simulation"]["duration_s"]/dt)+1
    rng = np.random.default_rng(c["simulation"]["random_seed"])
    arm = TwoJointFourMuscleArm(c["arm"]); physical = arm.initial.copy()
    desired = DesiredStateNode([c["target"]["x_m"], c["target"]["y_m"]])
    inverse = InverseModel(arm, c["inverse_model"])
    q_des = inverse.desired_joints(desired.wrist_xy, c["target"]["elbow_solution"])
    sensory = SensoryFeedbackNode(arm, dt, c["sensory_feedback"]["velocity_smoothing"])
    comparator = GoalComparator(c["goal"]["tolerance_m"], c["goal"]["hold_time_s"], dt)
    use_internal = c["internal_model"]["enabled"] if internal_override is None else internal_override
    forward = ForwardModel(arm, physical, c["internal_model"]["sensory_correction_gain"])
    motor_delay = DelayLine(c["delays"]["motor_s"], dt, np.zeros(4))
    feedback_delay = DelayLine(c["delays"]["sensory_s"], dt, physical.q)
    motor_sd = np.asarray(c["noise"]["motor_command_sd"], float)
    angle_sd = np.asarray(c["noise"]["joint_angle_feedback_sd_rad"], float)
    enabled_noise = c["noise"]["enabled"]

    keys = ["q","q_feedback","q_pred","wrist","wrist_feedback","wrist_pred","activation","command"]
    out = {k: [] for k in keys}; out.update(t=[], error=[], achieved=[])
    sensed_q, sensed_dq = physical.q.copy(), physical.dq.copy()
    estimated_q, estimated_dq = sensed_q.copy(), sensed_dq.copy()
    pred_xy = arm.wrist(physical.q)

    for i in range(n):
        clean = inverse.motor_command(q_des, estimated_q, estimated_dq)
        noisy = clean + (rng.normal(0, motor_sd, 4) if enabled_noise else 0)
        physical = arm.step(physical, motor_delay.push(np.clip(noisy, 0, 1)), dt)

        angle_feedback = physical.q.copy()
        if enabled_noise: angle_feedback += rng.normal(0, angle_sd, 2)
        sensed_q, sensed_dq, sensed_xy = sensory.decode(feedback_delay.push(angle_feedback))

        if use_internal:
            predicted, pred_xy = forward.update(clean, sensed_q, sensed_dq, dt)
            estimated_q, estimated_dq = predicted.q, predicted.dq
        else:
            predicted = physical.copy()  # only for array/animation shape; not used by control
            predicted.q, predicted.dq = sensed_q.copy(), sensed_dq.copy()
            pred_xy = sensed_xy.copy()
            estimated_q, estimated_dq = sensed_q, sensed_dq

        # Goal completion deliberately uses actual sensory feedback in both modes.
        error, achieved = comparator.update(desired.wrist_xy, sensed_xy)
        out["t"].append(i*dt); out["q"].append(physical.q.copy())
        out["q_feedback"].append(sensed_q.copy()); out["q_pred"].append(predicted.q.copy())
        out["wrist"].append(arm.wrist(physical.q)); out["wrist_feedback"].append(sensed_xy)
        out["wrist_pred"].append(pred_xy); out["activation"].append(physical.activation.copy())
        out["command"].append(clean); out["error"].append(error); out["achieved"].append(achieved)

    for k in out: out[k] = np.asarray(out[k])
    out.update(arm=arm, target=desired.wrist_xy, q_des=q_des, internal=use_internal)
    return out


def animate(r, c, save=None):
    arm, t = r["arm"], r["t"]; stride=max(1, round(c["animation"]["interval_s"]/c["simulation"]["dt_s"]))
    frames=np.arange(0,len(t),stride)
    fig,(ax,ax_e)=plt.subplots(1,2,figsize=(11,5.5)); reach=arm.l1+arm.l2
    ax.set(xlim=(-reach*1.1,reach*1.1),ylim=(-reach*1.1,reach*1.1),aspect="equal",xlabel="x (m)",ylabel="y (m)",title="Two-joint, four-muscle arm")
    ax.grid(alpha=.25); ax.plot(*r["target"],"r*",ms=16,label="Desired wrist")
    body,=ax.plot([],[],"o-",lw=6,label="Arm"); feedback,=ax.plot([],[],"s",label="Actual sensory feedback")
    prediction,=ax.plot([],[],"^",label="Forward prediction"); trail,=ax.plot([],[],alpha=.3)
    note=ax.text(.02,.98,"",transform=ax.transAxes,va="top"); ax.legend(loc="lower left",fontsize=8)
    ax_e.set(xlim=(0,t[-1]),ylim=(0,max(.02,r["error"].max()*1.05)),xlabel="time (s)",ylabel="desired - actual feedback (m)",title="Goal comparator")
    ax_e.grid(alpha=.25); errline,=ax_e.plot([],[],color="tab:red"); ax_e.axhline(c["goal"]["tolerance_m"],ls="--",color="gray")
    def update(j):
        i=frames[j]; s,e,w=arm.points(r["q"][i]); body.set_data([s[0],e[0],w[0]],[s[1],e[1],w[1]])
        feedback.set_data([r["wrist_feedback"][i,0]],[r["wrist_feedback"][i,1]])
        prediction.set_data([r["wrist_pred"][i,0]],[r["wrist_pred"][i,1]])
        prediction.set_visible(r["internal"]); trail.set_data(r["wrist"][:i+1,0],r["wrist"][:i+1,1])
        errline.set_data(t[:i+1],r["error"][:i+1]); note.set_text(f"t={t[i]:.2f} s\ninternal model={'ON' if r['internal'] else 'OFF'}\ngoal={'ACHIEVED' if r['achieved'][i] else 'not yet'}")
        return body,feedback,prediction,trail,errline,note
    ani=FuncAnimation(fig,update,frames=len(frames),interval=1000*c["animation"]["interval_s"],blit=False)
    fig.tight_layout()
    if save:
        if Path(save).suffix.lower() != ".gif": raise ValueError("This version saves GIF files; use a .gif name.")
        ani.save(save,writer=PillowWriter(fps=round(1/c["animation"]["interval_s"])),dpi=c["animation"]["dpi"])
    else: plt.show()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="architecture_config.json"); p.add_argument("--save"); p.add_argument("--no-animation",action="store_true")
    g=p.add_mutually_exclusive_group(); g.add_argument("--internal-model",action="store_true"); g.add_argument("--no-internal-model",action="store_true"); a=p.parse_args()
    override=True if a.internal_model else False if a.no_internal_model else None
    c=load(a.config); r=simulate(c,override)
    reached=np.flatnonzero(r["achieved"]); print("internal model:",r["internal"]); print("final comparator error (cm):",100*r["error"][-1]); print("goal achieved:",bool(reached.size));
    if reached.size: print("first sustained achievement (s):",r["t"][reached[0]])
    if not a.no_animation: animate(r,c,a.save)
if __name__=="__main__": main()

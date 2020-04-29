#!/usr/bin/env python3
"""
One leg swing model

Hip is fixed.
Two joints: hip and knee
Four muscles: GLU, HFL, RF, HAM

Author: Guoping Zhao, Lauflabor, gpzhaome@gmail.com
20180625
"""

import math
import time
import os
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer, functions

import mujoco_py

import matplotlib.pyplot as plt

from muscle_model.mtcmodel import MuscleTendonComplex

MODEL_LEG_XML = """
<?xml version="1.0" ?>
<mujoco model="leg">
  <compiler coordinate="local" inertiafromgeom="false"/>
  <custom>
    <numeric data="2" name="frame_skip"/>
  </custom>
  <default>
    <joint damping="0.0"/>
    <geom contype="0" friction="0.0 0.0 0.0" rgba="0.7 0.7 0 1"/>
  </default>
  <option gravity="1e-7 0 -9.81" integrator="RK4" timestep="0.0005"/>
  <size nstack="3000"/>
  <worldbody>
    <geom name="floor" pos="0 0 -3.0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
    <body name="cart" pos="0 0 0">
      <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
      <body name="thigh" pos="0 0 0">
        <inertial pos="0 0 0.3" mass="8.5" diaginertia="0.15 0.15 0.03"/>
        <joint axis="0 1 0" damping="0.0" name="hip" pos="0 0 0" type="hinge"/>
        <geom fromto="0 0 0 0 0 0.5" name="cpole" rgba="0 0.7 0.7 1" size="0.045 0.25" type="capsule"/>
        <body name="shank" pos="0 0 0.5">
          <inertial pos="0 0 0.3" mass="3.5" diaginertia="0.05 0.05 0.003"/>
          <joint axis="0 1 0" damping="0.0" name="knee" pos="0 0 0" type="hinge"/>
          <geom fromto="0 0 0 0 0 0.5" name="cpole2" rgba="0.7 0.2 0.2 1" size="0.045 0.25" type="capsule"/>

        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-10000 10000" gear="1" joint="hip" name="hipMotor"/>
    <motor ctrllimited="true" ctrlrange="-10000 10000" gear="1" joint="knee" name="kneeMotor"/>
  </actuator>
</mujoco>
"""

#           <site name="tip" pos="0 0 .5" size="0.01 0.01"/>

model = load_model_from_xml(MODEL_LEG_XML)
sim = MjSim(model)
viewer = MjViewer(sim)

qpos = np.zeros(2)
qvel = np.zeros(2)
old_state = sim.get_state()
angHip = 120.0 / 180 * np.pi
angKne = (180.0 - 40.0) / 180 * np.pi
qpos[0] = angHip  # hip joint initial angle
qpos[1] = np.pi - angKne  # knee joint initial angle
qvel[0] = 0.0  # hip joint initial velocity
qvel[1] = 0.0  # knee joint initial velocity
new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                 old_state.act, old_state.udd_state)
sim.set_state(new_state)
sim.forward()

# viewer.render()

# create muscle
timestep = 5e-4
# Series elastic element (SE) force-length relationship
eref = 0.04     # [lslack] tendon reference strain
# excitation-contraction coupling
preAct = 0.01     # [] preactivation
tau = 0.01    # [s] delay time constant
# contractile element (CE) force-length relationship
w = 0.56   # [lopt] width
c = 0.05   # []; remaining force at +/- width
# CE force-velocity relationship
N = 1.5    # Fmax] eccentric force enhancement
K = 5.0    # [] shape factor
stim = 0.0  # initial stimulation
vce = 0.0
frcmtc = 0.0

# GLU, gluteus maximus
frcmax = 1500.0   # maximum isometric force [N]
lopt = 0.11     # optimum fiber length CE [m]
vmax = 12.0       # maximum contraction velocity [lopt/s]
lslack = 0.13   # tendon slack length [m]
# level arm and reference angle
r       =  np.array((0.08,))  # [m]   constant lever contribution
phiref  = np.array((120*np.pi/180,))   # [rad] reference angle at which MTU length equals
phimaxref = np.array((0.0, 0.0))
rho     = np.array((0.5,))  #       sum of lopt and lslack
dirAng = np.array((-1.0,))  # angle increase leads to MTC length decrease
offsetCorr = np.array((0,))  # no level arm correction
# typeMuscle = 1  # monoarticular
phiScale = np.array((0.0,))

act = preAct
lmtc = 0.0  # will be computed in the initialization
lce = lopt  # will be computed in the initialization

paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
musGLU = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle, paraMusAttach=paraMusAttach,
                             offsetCorr=offsetCorr, timestep=timestep, nameMuscle="GLU",
                             angJoi=np.array((angHip,)))

# HFL, hip flexor
frcmax   = 2000  # maximum isometric force [N]
lopt   = 0.11  # optimum fiber length CE [m]
vmax   =   12.0  # maximum contraction velocity [lopt/s]
lslack = 0.10  # tendon slack length [m]
# level arm and reference angle
r       = np.array((0.08,))  # [m]   constant lever contribution
phiref  = np.array((160*np.pi/180,))  # [rad] reference angle at which MTU length equals
phimaxref = np.array((0.0, 0.0))
rho     = np.array((0.5,))  # sum of lopt and lslack
dirAng = np.array((1.0,))  # angle increase leads to MTC length increase
offsetCorr = np.array((0,))  # no level arm correction
# typeMuscle = 1  # monoarticular
phiScale = np.array((0.0,))

# act = preAct
act = 0.0
lmtc = 0.0  # should be computed based on the joint angle and joint geometry
lce = lopt

paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
musHFL = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle, paraMusAttach=paraMusAttach,
                             offsetCorr=offsetCorr, timestep=timestep, nameMuscle="HFL",
                             angJoi=np.array((angHip,)))

# HAM, hamstring
frcmax   = 3000  # maximum isometric force [N]
lopt   = 0.10  # optimum fiber length CE [m]
vmax   =   12.0  # maximum contraction velocity [lopt/s]
lslack = 0.31  # tendon slack length [m]
# hamstring hip level arm and refernce angle
rHAMh       = 0.08  # [m]   constant lever contribution
phirefHAMh  = 150*np.pi/180    # [rad] reference angle at which MTU length equals
rhoHAMh     = 0.5   #      sum of lopt and lslack
# hamstring knee level arm and reference angle
rHAMk       = 0.05  # [m]   constant lever contribution
phirefHAMk  = 180*np.pi/180    # [rad] reference angle at which MTU length equals
rhoHAMk     = 0.5   # sum of lopt and lslack

r = np.array((rHAMh, rHAMk))
phiref = np.array((phirefHAMh, phirefHAMk))
phimaxref = np.array((0.0, 0.0))
rho = np.array((rhoHAMh, rhoHAMk))
dirAng = np.array((-1.0, 1.0))
offsetCorr = np.array((0, 0))
# typeMuscle = 2
phiScale = np.array((0.0, 0.0))

act = preAct
lmtc = 0.0
lce = lopt

paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
musHAM = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle, paraMusAttach=paraMusAttach,
                             offsetCorr=offsetCorr, timestep=timestep, nameMuscle="HAM",
                             angJoi=np.array((angHip, angKne)))


# rectus femoris, REF
frcmax   = 1200  # maximum isometric force [N]
lopt   = 0.08  # optimum fiber length CE [m]
vmax   =   12.0  # maximum contraction velocity [lopt/s]
lslack = 0.35  # tendon slack length [m]
# REF group attachement (hip)
rREFh      =       0.08  # [m]   constant lever contribution
phirefREFh = 170*np.pi/180  # [rad] reference angle at which MTU length equals
rhoREFh    =        0.3  # sum of lopt and lslack
# REF group attachement (knee)
rREFkmax     = 0.06     # [m]   maximum lever contribution
rREFkmin     = 0.04     # [m]   minimum lever contribution
phimaxREFk   = 165*np.pi/180   # [rad] angle of maximum lever contribution
phiminREFk   =  45*np.pi/180   # [rad] angle of minimum lever contribution
phirefREFk   = 125*np.pi/180   # [rad] reference angle at which MTU length equals
rhoREFk      = 0.5          # sum of lopt and lslack
phiScaleREFk = np.arccos(rREFkmin/rREFkmax)/(phiminREFk-phimaxREFk)

r = np.array((rREFh, rREFkmax))
phiref = np.array((phirefREFh, phirefREFk))
phimaxref = np.array((0.0, phimaxREFk))
rho = np.array((rhoREFh, rhoREFk))
dirAng = np.array((1.0, -1.0))
offsetCorr = np.array((0, 1))
phiScale = np.array((0.0, phiScaleREFk))

act = preAct
lmtc = 0.0
lce = lopt

paraMuscle = [frcmax, vmax, eref, lslack, lopt, tau, w, c, N, K]
stateMuscle = [stim, act, lmtc, lce, vce, frcmtc]
paraMusAttach = [r, phiref, phimaxref, rho, dirAng, phiScale]
musREF = MuscleTendonComplex(paraMuscle=paraMuscle, stateMuscle=stateMuscle, paraMusAttach=paraMusAttach,
                             offsetCorr=offsetCorr, timestep=timestep, nameMuscle="REF",
                             angJoi=np.array((angHip, angKne)))




nFrame = int(0.6/timestep)
qposAll = np.zeros((nFrame, 2))
qvelAll = np.zeros((nFrame, 2))
torAll = np.zeros((nFrame, 2))
frcMusAll = np.zeros((nFrame, 4))
actMusAll = np.zeros((nFrame, 4))

sys_state = sim.get_state()
step = 0

startTime = time.process_time()

while True:
    t = time.time()

    actMusAll[step, :] = np.array((musGLU.act, musHFL.act, musHAM.act, musREF.act))
    actMusAll[step, 0] = musREF.levelArm[1]
    torAll[step, :] = sim.data.qfrc_actuator
    frcMusAll[step, :] = np.array((musGLU.frcmtc, musHFL.frcmtc, musHAM.frcmtc, musREF.frcmtc))
    # qposAll[step, :] = sys_state.qpos
    qposAll[step, :] = np.array((angHip, angKne))
    qvelAll[step, :] = sys_state.qvel

    # sim.data.ctrl[0] = 0.0
    # sim.data.ctrl[1] = 0.0
    # sim.data.ctrl[0] = musGLU.frcmtc * musGLU.levelArm
    # sim.data.ctrl[0] = - musREF.frcmtc * musREF.levelArm[0]
    # sim.data.ctrl[1] = - musREF.frcmtc * musREF.levelArm[1]
    sim.data.ctrl[0] = musGLU.frcmtc * musGLU.levelArm - musHFL.frcmtc * musHFL.levelArm + \
                       musHAM.frcmtc * musHAM.levelArm[0] - musREF.frcmtc * musREF.levelArm[0]
    sim.data.ctrl[1] = musHAM.frcmtc * musHAM.levelArm[1] - musREF.frcmtc * musREF.levelArm[1]

    sim.data.qfrc_applied[:] = np.zeros(2)

    # # sim_state = sim.get_state()
    # if step == 100:
    #     old_state = sim.get_state()
    #     qpos[0] = 170.0 / 180 * np.pi
    #     qpos[1] = 0.0
    #     qvel = 0
    #     new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
    #                                      old_state.act, old_state.udd_state)
    #     sim.set_state(new_state)
    #     sim.forward()
    #
    # if 200 < step < 210:
    #     sim.data.ctrl[1] = 0.5
    #     sim.data.qfrc_applied[1] = 50

    sim.step()

    step += 1

    sys_state = sim.get_state()
    angHip, angKne = sys_state.qpos
    angKne = np.pi - angKne

    # musGLU.lmtc = lslack + lopt - (angHip - phirefGLU) * rGLU * rhoGLU
    musGLU.stim = 0.1
    musGLU.stepUpdateState(np.array((angHip,)))
    musHFL.stim = 0.1
    musHFL.stepUpdateState(np.array((angHip,)))
    musHAM.stim = 0.1
    musHAM.stepUpdateState(np.array((angHip, angKne)))
    musREF.stim = 0.1
    musREF.stepUpdateState(np.array((angHip, angKne)))

    # actAll[step] = musGLU.act

    # get the sum of force/torque acting on the joint, needs to call mucojo function mj_inverse, otherwise qfrc_inverse
    # will be all zeros, call mj_inverseSkip to improve speed
    # functions.mj_inverse(sim.model, sim.data)
    # torAll[step, :] = sim.data.qfrc_inverse

    # torAll[step, :] = sim.data.qfrc_passive  # joint passive force/torque, created by joint damping/friction
    # torAll[step, :] = sim.data.qfrc_actuator

    # x, y = math.cos(t), math.sin(t)
    # viewer.add_marker(pos=np.array([x, y, 1]),
    #                   label=str(t))

    viewer.render()

    # if step > 100 and os.getenv('TESTING') is not None:
    #     break

    if step >= nFrame:
        break

endTime = time.process_time()
timeElap = endTime - startTime
print(timeElap)


fig, axes = plt.subplots(nrows=1, ncols=3)
line1, = axes[0].plot(qposAll[:, 0] * 180.0 / np.pi, label='shoulder')
line2, = axes[0].plot(qposAll[:, 1] * 180.0 / np.pi, label='elbow')
# axes[0].legend()
axes[0].grid(True)
torLine, = axes[1].plot(torAll[:, 0])
torLine, = axes[1].plot(torAll[:, 1], linestyle='--')
actLine, = axes[2].plot(actMusAll[:, 0])

plt.show()
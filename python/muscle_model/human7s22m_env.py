#!/usr/bin/env python3
"""
Human locomotion model with 7 segments and 11 muscles per leg

Author: Guoping Zhao, Lauflabor, gpzhaome@gmail.com
All rights reserved
20180701

"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env
import random
import math
import matplotlib.pyplot as plt
import time
import os
import scipy.io
import numpy as np
from mujoco_py import load_model_from_xml, MjSim, MjViewer, functions, load_model_from_path

import mujoco_py
# import quaternion

import matplotlib.pyplot as plt

import sys
from thesis_galljamov18.python.muscle_model import humanmuscle

# import humanmuscle as hummus

from os import path
import six

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 1000


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class Human7s22mEnv(gym.Env, utils.EzPickle):  #Human7s22mEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mat = scipy.io.loadmat('Trajectory/Traj_Step_old_Fil.mat')
        self.mat = mat['Data']
        # mat_dist = scipy.io.loadmat('Trajectory/Phi_Traj_Step_12.mat')
        # self.Dist = mat_dist['Dist']
        # self.mu_r = self.Dist[0][0]
        # self.mu_l = self.Dist[1][0]
        # self.sigma_r = self.Dist[2][0]
        # self.sigma_l = self.Dist[3][0]
        self.max_traj = np.zeros(29)
        self.x_dist = 0

        self.data_length = 426
        for i in range(29):
            for j in range(self.data_length):
                Traj = self.mat[j][0]
                self.max_traj[i] = max(self.max_traj[i], max(np.absolute(Traj[i])))

        for i in range(29):
            if self.max_traj[i] == 0:
                self.max_traj[i] =1

        self.v_max = 2#self.max_traj[15]
        self.traj_id = 0
        self.phi = 0
        self.Traj = self.mat[self.traj_id][0]
        self.traj_length = np.shape(self.Traj)[1]
        self.target_vel = self.Traj[:,self.traj_length-1][15:18]#np.array(np.mean(self.Traj[15:18,:], axis =1))
        self.step_no = 0
        # self.start_no =self.step_no
        self.err_limit = 14
        self.alive_bonus = 5.0
        self.max_frc = 2000
        model_path = 'human7segment.xml'
        frame_skip = 5

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)

        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'videos.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # ---------------------- initialize the model ----------------------------
        # self.ixqposTru = list(range(0, 3))
        # self.ixqqutTru = list(range(3, 7))
        # self.ixqHipFroR, self.ixqHipSagR, self.ixqHipVerR, self.ixqKneR, self.ixqAnkR, self.ixqHipFroL, self.ixqHipSagL, self.ixqHipVerL, self.ixqKneL, self.ixqAnkL = list(
        #     range(7, 17))

        # # set initial joint angle
        # # trunk CoM initial height
        # # posTrunk = self.init_qpos[self.ixqposTru]
        # # posTru = np.array((0.0, 0.0, 1.02))
        # # trunk angle, lean a bit forward
        # vecRot = np.array((0, 1, 0))
        # angRot = 11.0 / 180 * np.pi  # 10 degree
        # qutTru = np.append(np.cos(angRot / 2), vecRot * np.sin(angRot / 2))
        # # qutTru = quaternion.as_quat_array(qutRot)


        self.init_qpos[0:3] = self.Traj[:,0][0:3]
        self.init_qpos[3:15] = self.Traj[:,0][3:15]
        self.init_qvel[0:14] = self.Traj[:,0][15:29]

        angHipFroR, angHipSagR, angKneR, angAnkR, angHipFroL, angHipSagL, angKneL, angAnkL = self.init_qpos[7:15]
        angHipAbdR = -angHipFroR
        angHipAbdL = angHipFroL
        angHipSagR = angHipSagR + np.pi
        angHipSagL = angHipSagL + np.pi
        angKneR = np.pi - angKneR
        angKneL = np.pi - angKneL
        angAnkR = angAnkR + np.pi / 2.0
        angAnkL = angAnkL + np.pi / 2.0
        # leva qvel as default
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, self.init_qpos, self.init_qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

        # create 11 muscles for each leg
        # NOTE: the timestep has to be the same as in the xml file
        timestep = 5e-4
        humanmuscle.timestep = timestep
        nMus = 22
        self.musHABR = humanmuscle.HAB(angHipAbdR)
        self.musHADR = humanmuscle.HAD(angHipAbdR)
        self.musGLUR = humanmuscle.GLU(angHipSagR)
        self.musHFLR = humanmuscle.HFL(angHipSagR)
        self.musHAMR = humanmuscle.HAM(angHipSagR, angKneR)
        self.musREFR = humanmuscle.REF(angHipSagR, angKneR)
        self.musVASR = humanmuscle.VAS(angKneR)
        self.musBFSHR = humanmuscle.BFSH(angKneR)
        self.musGASR = humanmuscle.GAS(angKneR, angAnkR)
        self.musSOLR = humanmuscle.SOL(angAnkR)
        self.musTIAR = humanmuscle.TIA(angAnkR)
        self.musHABL = humanmuscle.HAB(angHipAbdL)
        self.musHADL = humanmuscle.HAD(angHipAbdL)
        self.musGLUL = humanmuscle.GLU(angHipSagL)
        self.musHFLL = humanmuscle.HFL(angHipSagL)
        self.musHAML = humanmuscle.HAM(angHipSagL, angKneL)
        self.musREFL = humanmuscle.REF(angHipSagL, angKneL)
        self.musVASL = humanmuscle.VAS(angKneL)
        self.musBFSHL = humanmuscle.BFSH(angKneL)
        self.musGASL = humanmuscle.GAS(angKneL, angAnkL)
        self.musSOLL = humanmuscle.SOL(angAnkL)
        self.musTIAL = humanmuscle.TIA(angAnkL)

        self.frcmtc_buffer = np.zeros((22,7))
        self.vce_buffer = np.zeros((22,7))
        self.lce_buffer = np.array([[self.musHABR.lce, self.musHADR.lce, self.musGLUR.lce, self.musHFLR.lce, self.musHAMR.lce, self.musREFR.lce,\
                          self.musBFSHR.lce,self.musVASR.lce, self.musGASR.lce, self.musSOLR.lce, self.musTIAR.lce, self.musHABL.lce, self.musHADL.lce,\
                          self.musGLUL.lce, self.musHFLL.lce, self.musHAML.lce,self.musREFL.lce, self.musBFSHL.lce, self.musVASL.lce, self.musGASL.lce,\
                          self.musSOLL.lce, self.musTIAL.lce] for i in range(7)]).transpose()
        # -------------- run step ----------------------
        observation, _reward, done, _info = self.step(np.zeros(nMus))
        # observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        #assert not done
        self.obs_dim = observation.size

        print(self.obs_dim)

        # -------------- set actuator control range ---------------
        # bounds = self.model.actuator_ctrlrange.copy()
        # self.bounds = bounds
        # low = bounds[:, 0]
        # high = bounds[:, 1]
        # self.action_space = spaces.Box(low=low, high=high)
        # -------------- set muscle stimulation range ---------------
        low = np.zeros((nMus))
        high = np.ones((nMus))
        self.action_space = spaces.Box(low=low, high=high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.seed()

        utils.EzPickle.__init__(self)


    def _get_obs(self):
        # TODO: add muscle info if needed
        # data = self.sim.data
        # data_pv =  np.concatenate([data.qpos.flat[1:15], data.qvel.flat])
        # data_pv =  data_pv / (1 * self.max_traj[1:29])
        # leg_frc_r =  data.cfrc_ext[4] / self.max_frc
        # leg_frc_l =  data.cfrc_ext[7] / self.max_frc
        # vel_target =  self.target_vel / self.v_max
        # frcmtc = [self.musHABR.frcmtc/3000, self.musHADR.frcmtc/4500, self.musGLUR.frcmtc/1500, self.musHFLR.frcmtc/2000, self.musHAMR.frcmtc/3000, \
        #          self.musREFR.frcmtc/1200, self.musBFSHR.frcmtc/350, self.musVASR.frcmtc/6000, self.musGASR.frcmtc/1500, self.musSOLR.frcmtc/4000, \
        #          self.musTIAR.frcmtc/800, \
        #          self.musHABL.frcmtc/3000, self.musHADL.frcmtc/4500, self.musGLUL.frcmtc/1500, self.musHFLL.frcmtc/2000, self.musHAML.frcmtc/3000, \
        #          self.musREFL.frcmtc/1200, self.musBFSHL.frcmtc/350, self.musVASL.frcmtc/6000, self.musGASL.frcmtc/1500, self.musSOLL.frcmtc/4000, \
        #          self.musTIAL.frcmtc/800]
        # self.buffer = np.roll(self.buffer,1)
        # self.buffer[:,0] = frcmtc
        # frcmtc = self.buffer[:,6]
        # data = np.concatenate([data_pv.flat, leg_frc_r.flat, leg_frc_l.flat, frcmtc, vel_target.flat]) #
        # #data = np.append(data, self.phi/10)
        # return  np.clip(data, -10, 10 )

        data = self.sim.data
        data_pv =  np.concatenate([data.qpos.flat[1:15], data.qvel.flat])
        leg_frc_r =  data.cfrc_ext[4]
        leg_frc_l =  data.cfrc_ext[7]
        vel_target =  self.target_vel/2

        frcmtc = [self.musHABR.frcmtc, self.musHADR.frcmtc, self.musGLUR.frcmtc, self.musHFLR.frcmtc, self.musHAMR.frcmtc, self.musREFR.frcmtc,\
                  self.musBFSHR.frcmtc, self.musVASR.frcmtc, self.musGASR.frcmtc, self.musSOLR.frcmtc, self.musTIAR.frcmtc,\
                  self.musHABL.frcmtc, self.musHADL.frcmtc, self.musGLUL.frcmtc, self.musHFLL.frcmtc, self.musHAML.frcmtc, \
                  self.musREFL.frcmtc, self.musBFSHL.frcmtc, self.musVASL.frcmtc, self.musGASL.frcmtc, self.musSOLL.frcmtc, self.musTIAL.frcmtc]

        lce = [self.musHABR.lce, self.musHADR.lce, self.musGLUR.lce, self.musHFLR.lce, self.musHAMR.lce, self.musREFR.lce,\
                  self.musBFSHR.lce, self.musVASR.lce, self.musGASR.lce, self.musSOLR.lce, self.musTIAR.lce,\
                  self.musHABL.lce, self.musHADL.lce, self.musGLUL.lce, self.musHFLL.lce, self.musHAML.lce, \
                  self.musREFL.lce, self.musBFSHL.lce, self.musVASL.lce, self.musGASL.lce, self.musSOLL.lce, self.musTIAL.lce]

        vce = [self.musHABR.vce, self.musHADR.vce, self.musGLUR.vce, self.musHFLR.vce, self.musHAMR.vce, self.musREFR.vce,\
                  self.musBFSHR.vce, self.musVASR.vce, self.musGASR.vce, self.musSOLR.vce, self.musTIAR.vce,\
                  self.musHABL.vce, self.musHADL.vce, self.musGLUL.vce, self.musHFLL.vce, self.musHAML.vce, \
                  self.musREFL.vce, self.musBFSHL.vce, self.musVASL.vce, self.musGASL.vce, self.musSOLL.vce, self.musTIAL.vce]

        act = [self.musHABR.act, self.musHADR.act, self.musGLUR.act, self.musHFLR.act, self.musHAMR.act, self.musREFR.act,\
                  self.musBFSHR.act, self.musVASR.act, self.musGASR.act, self.musSOLR.act, self.musTIAR.act,\
                  self.musHABL.act, self.musHADL.act, self.musGLUL.act, self.musHFLL.act, self.musHAML.act, \
                  self.musREFL.act, self.musBFSHL.act, self.musVASL.act, self.musGASL.act, self.musSOLL.act, self.musTIAL.act]

        self.frcmtc_buffer = np.roll(self.frcmtc_buffer,1)
        self.frcmtc_buffer[:,0] = frcmtc
        frcmtc = self.frcmtc_buffer[:,6]
        self.lce_buffer = np.roll(self.lce_buffer,1)
        self.lce_buffer[:,0] = lce
        lce = self.lce_buffer[:,6]
        self.vce_buffer = np.roll(self.vce_buffer,1)
        self.vce_buffer[:,0] = vce
        vce = self.vce_buffer[:,6]
        #lce[8] =min(lce[8],0.07)
        #lce[19] =min(lce[19],0.07)
        data = np.concatenate([data_pv.flat, leg_frc_r.flat, leg_frc_l.flat, frcmtc, lce, vce, act, vel_target.flat]) #
        return data


    def step(self, stimu):
        # clip stimu between 0 and 1
        # stimu = np.ones(22)*.01
        stimu = np.clip(stimu, 0.001, 1)
        # if(self.step_no == self.start_no):
        #     self.reset_muscle(stimu)
        #traj_id = step_in[22]
        #step_no = int(step_in[23])
        #print(stimu, traj_id, step_no)
        #pos_before = mass_center(self.model, self.sim)
        #qpos_before = self.sim.data.qpos

        self.do_simulation(stimu, self.frame_skip)

        #pos_after = mass_center(self.model, self.sim)

        # TODO: modify the cost/reward function
        data = self.sim.data
        #lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        #lin_vel_cost = 1 * (pos_after - pos_before) / self.model.opt.timestep
        #quad_ctrl_cost = 0.1 * np.square(data.ctrl/500).sum()
        #quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        #quad_impact_cost = min(quad_impact_cost, 10)
        # reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        foot_vec = np.append((data.body_xpos[1] - data.body_xpos[4]), (data.body_xpos[1] - data.body_xpos[7]))
        efoot_vec =  30 * (np.subtract(self.Traj[:,self.step_no][29:35], foot_vec))   #40
        epos_com = 20 * (np.subtract((self.Traj[:,self.step_no][0:3] + [self.x_dist, 0, 0]), qpos[0:3]))
        epos_ang = 12 * (np.subtract(self.Traj[:,self.step_no][3:15], qpos[3:15]))
        # epos_ang[0:4] = 1 * (epos_ang[0:4])    #1
        evel_com =  (np.subtract(self.Traj[:,self.step_no][15:18], qvel[0:3]))
        evel_ang =  (np.subtract(self.Traj[:,self.step_no][18:29], qvel[3:14]))
        #evel_ang_l =  (np.subtract(self.Traj[:,self.step_no][26:29], qvel[11:14]))
        evel = np.hstack((evel_com, evel_ang))

        sq_efoot_vec = 1 * np.square(efoot_vec)
        sq_epos_com = 1 * np.square(epos_com[0:3])
        sq_epos_ang = 1 * np.square(epos_ang)
        sq_evel = 0.1 * np.square(evel)
        sq_evel[0:3] = 20 * sq_evel[0:3]
        trajerr_list = np.exp(-1* np.hstack(( sq_efoot_vec, sq_epos_com, sq_epos_ang , sq_evel))) #
        trajerr = np.sum(trajerr_list)
        death_err = np.sum(trajerr_list[6:35])
        #if self.step_no ==0:
            #print(data.body_xpos[1], data.body_xpos[4], data.body_xpos[7])

        self.step_no += 1

        if self.step_no == self.traj_length:
            self.x_dist += self.Traj[:,self.traj_length-1][0]
            self._get_traj_id()
            self.step_no = 0
            #print (self.traj_id)
        mus_stim =  np.sum(stimu)
        reward = (8*(trajerr - self.err_limit) / 11) + (2*(22-mus_stim)/22)  # self.alive_bonus  + (trajerr / 7) (1*(22-mus_stim)/22)
        # print(sim.data.body_xpos[7][1] - sim.data.body_xpos[4][1])
        done = bool((qpos[2] < 0.8) or (qpos[2] > 1.4) or (qpos[1] < -0.5) or (qpos[1] > 0.5) or ( death_err < self.err_limit))
        #print(self.step_no, ": ",self.musHABR.vce, self.musHABR.frcmtc )
        if done:
            # print(trajerr_list) #np.exp(-1*sq_efoot_vec)
            reward = 0
            #self.step_no = 0
            self.x_dist = 0

            self.phi = random.randrange(0,10)
            self.traj_id = random.randrange(420)
            self.Traj = self.mat[self.traj_id][0]
            self.traj_length = np.shape(self.Traj)[1]
            self.step_no = int(np.round(self.phi * self.traj_length / 10))
            # self.start_no =self.step_no
            self.target_vel = self.Traj[:,self.traj_length-1][15:18]#np.array(np.mean(self.Traj[15:18,:], axis =1))
            # self.init_qpos[1:15] = np.random.normal(mu[:,self.phi][1:15], 0.3*sigma[:,self.phi][1:15], 14)   #self.Traj[:,0][0:3]
            # self.init_qvel[0:14] = np.random.normal(mu[:,self.phi][15:29], 0.3*sigma[:,self.phi][15:29], 14)
            self.init_qpos[0:15] = self.Traj[:,self.step_no][0:15]
            self.init_qvel[0:14] = self.Traj[:,self.step_no][15:29]

            self.reset_model()
            self.reset_muscle()
        state_ob = self._get_obs()
        return state_ob, reward, done, dict(reward=reward)

    def _get_traj_id(self):
        max_err = 0
        traj_id = 0
        data = self.sim.data
        qpos = data.qpos
        qvel = data.qvel
        foot_vec = np.append((data.body_xpos[1] - data.body_xpos[4]), (data.body_xpos[1] - data.body_xpos[7]))
        if self.traj_id % 2 == 0:
            start = 1
        else:
            start = 0
        for i in range(start, 420,2):
            Traj =  self.mat[i][0]
            # l = np.shape(Traj)[1]
            efoot_vec =  30 * (np.subtract(Traj[:,0][29:35], foot_vec))
            epos_com = 20 * (np.subtract(Traj[:,0][1:3], qpos[1:3]))
            epos_ang = 12 * (np.subtract(Traj[:,0][3:15], qpos[3:15]))
            evel =  (np.subtract(Traj[:,0][15:29], qvel))
            sq_efoot_vec = 1 * np.square(efoot_vec)
            sq_epos_com = 1 * np.square(epos_com)
            sq_epos_ang = 1 * np.square(epos_ang)
            # sq_epos_ang[7] = 1* sq_epos_ang[7]
            # sq_epos_ang[11] = 1* sq_epos_ang[11]
            sq_evel = 0.1 * np.square(evel)
            sq_evel[0:3] = 30 * sq_evel[0:3]
            trajerr = np.exp(-1* np.hstack(( sq_efoot_vec, sq_epos_com, sq_epos_ang , sq_evel))) #
            trajerr=np.sum(trajerr)
            if trajerr > max_err:
                traj_id = i
                max_err = trajerr
        self.phi = 0
        self.traj_id = traj_id
        self.Traj = self.mat[self.traj_id][0]
        self.traj_length = np.shape(self.Traj)[1]
        self.target_vel = self.Traj[:,self.traj_length-1][15:18]#np.array(np.mean(self.Traj[15:18,:], axis =1))

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos ,#+ self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel #+ self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        # TODO: add reset for muscle property

        return self._get_obs()

    def reset_muscle(self):

        angHipFroR, angHipSagR, angKneR, angAnkR, angHipFroL, angHipSagL, angKneL, angAnkL = self.init_qpos[7:15]
        angHipAbdR = -angHipFroR
        angHipAbdL = angHipFroL
        angHipSagR = angHipSagR + np.pi
        angHipSagL = angHipSagL + np.pi
        angKneR = np.pi - angKneR
        angKneL = np.pi - angKneL
        angAnkR = angAnkR + np.pi / 2.0
        angAnkL = angAnkL + np.pi / 2.0

        timestep = 5e-4
        humanmuscle.timestep = timestep

        self.musHABR = humanmuscle.HAB(angHipAbdR)
        self.musHADR = humanmuscle.HAD(angHipAbdR)
        self.musGLUR = humanmuscle.GLU(angHipSagR)
        self.musHFLR = humanmuscle.HFL(angHipSagR)
        self.musHAMR = humanmuscle.HAM(angHipSagR, angKneR)
        self.musREFR = humanmuscle.REF(angHipSagR, angKneR)
        self.musVASR = humanmuscle.VAS(angKneR)
        self.musBFSHR = humanmuscle.BFSH(angKneR)
        self.musGASR = humanmuscle.GAS(angKneR, angAnkR)
        self.musSOLR = humanmuscle.SOL(angAnkR)
        self.musTIAR = humanmuscle.TIA(angAnkR)
        self.musHABL = humanmuscle.HAB(angHipAbdL)
        self.musHADL = humanmuscle.HAD(angHipAbdL)
        self.musGLUL = humanmuscle.GLU(angHipSagL)
        self.musHFLL = humanmuscle.HFL(angHipSagL)
        self.musHAML = humanmuscle.HAM(angHipSagL, angKneL)
        self.musREFL = humanmuscle.REF(angHipSagL, angKneL)
        self.musVASL = humanmuscle.VAS(angKneL)
        self.musBFSHL = humanmuscle.BFSH(angKneL)
        self.musGASL = humanmuscle.GAS(angKneL, angAnkL)
        self.musSOLL = humanmuscle.SOL(angAnkL)
        self.musTIAL = humanmuscle.TIA(angAnkL)

        self.frcmtc_buffer = np.zeros((22,7))
        self.vce_buffer = np.zeros((22,7))
        self.lce_buffer = np.array([[self.musHABR.lce, self.musHADR.lce, self.musGLUR.lce, self.musHFLR.lce, self.musHAMR.lce, self.musREFR.lce,\
                          self.musBFSHR.lce,self.musVASR.lce, self.musGASR.lce, self.musSOLR.lce, self.musTIAR.lce, self.musHABL.lce, self.musHADL.lce,\
                          self.musGLUL.lce, self.musHFLL.lce, self.musHAML.lce,self.musREFL.lce, self.musBFSHL.lce, self.musVASL.lce, self.musGASL.lce,\
                          self.musSOLL.lce, self.musTIAL.lce] for i in range(7)]).transpose()
        #self.musHABR.actsubstep, self.musHABR.lcesubstep, self.musHABR.vce  = [stimu[22], stimu[23], 0]
        # self.musHADR.actsubstep, self.musHADR.lcesubstep, self.musHADR.vce  = [stimu[24], stimu[25], 0]
        # self.musGLUR.actsubstep, self.musGLUR.lcesubstep, self.musGLUR.vce  = [stimu[26], stimu[27], 0]
        # self.musHFLR.actsubstep, self.musHFLR.lcesubstep, self.musHFLR.vce  = [stimu[28], stimu[29], 0]
        # self.musHAMR.actsubstep, self.musHAMR.lcesubstep, self.musHAMR.vce  = [stimu[30], stimu[31], 0]
        # self.musREFR.actsubstep, self.musREFR.lcesubstep, self.musREFR.vce  = [stimu[32], stimu[33], 0]
        # self.musVASR.actsubstep, self.musVASR.lcesubstep, self.musVASR.vce  = [stimu[34], stimu[35], 0]
        # self.musBFSHR.actsubstep, self.musBFSHR.lcesubstep, self.musBFSHR.vce  = [stimu[36], stimu[37], 0]
        # self.musGASR.actsubstep, self.musGASR.lcesubstep, self.musGASR.vce  = [stimu[38], stimu[39], 0]
        # self.musSOLR.actsubstep, self.musSOLR.lcesubstep, self.musSOLR.vce  = [stimu[40], stimu[41], 0]
        # self.musTIAR.actsubstep, self.musTIAR.lcesubstep, self.musTIAR.vce  = [stimu[42], stimu[43], 0]
        # self.musHABL.actsubstep, self.musHABL.lcesubstep, self.musHABL.vce  = [stimu[44], stimu[45], 0]
        # self.musHADL.actsubstep, self.musHADL.lcesubstep, self.musHADL.vce  = [stimu[46], stimu[47], 0]
        # self.musGLUL.actsubstep, self.musGLUL.lcesubstep, self.musGLUL.vce  = [stimu[48], stimu[49], 0]
        # self.musHFLL.actsubstep, self.musHFLL.lcesubstep, self.musHFLL.vce  = [stimu[50], stimu[51], 0]
        # self.musHAML.actsubstep, self.musHAML.lcesubstep, self.musHAML.vce  = [stimu[52], stimu[53], 0]
        # self.musREFL.actsubstep, self.musREFL.lcesubstep, self.musREFL.vce  = [stimu[54], stimu[55], 0]
        # self.musVASL.actsubstep, self.musVASL.lcesubstep, self.musVASL.vce  = [stimu[56], stimu[57], 0]
        # self.musBFSHL.actsubstep, self.musBFSHL.lcesubstep, self.musBFSHL.vce  = [stimu[58], stimu[59], 0]
        # self.musGASL.actsubstep, self.musGASL.lcesubstep, self.musGASL.vce  = [stimu[60], stimu[61], 0]
        # self.musSOLL.actsubstep, self.musSOLL.lcesubstep, self.musSOLL.vce  = [stimu[62], stimu[63], 0]
        # self.musTIAL.actsubstep, self.musTIAL.lcesubstep, self.musTIAL.vce  = [stimu[64], stimu[65], 0]


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        # self.viewer.cam.lookat[2] += .8
        self.viewer.cam.lookat[2] += 0
        self.viewer.cam.elevation = -20

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        old_viewer = self.viewer
        for v in self._viewers.values():
            self.viewer = v
            self.viewer_setup()
        self.viewer = old_viewer
        return ob

    # def reset_trajectory(self):
    #     self.sim.reset()
    #     ob = self.reset_model()
    #     old_viewer = self.viewer
    #     for v in self._viewers.values():
    #         self.viewer = v
    #         self.viewer_setup()
    #     self.viewer = old_viewer
    #     return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, stimu, n_frames):
        # stimu: muslce stimulation as action
        for _ in range(n_frames):
            # update muscle, NOTE, this is before the step simulation, it can also be after the step simulation
            sys_state = self.sim.get_state()
            qpos = sys_state.qpos
            # # qvel = sys_state.qvel
            angHipFroR, angHipSagR, angKneR, angAnkR, angHipFroL, angHipSagL, angKneL, angAnkL = qpos[7:15]
            angHipAbdR = -angHipFroR
            angHipAbdL = angHipFroL
            angHipSagR = angHipSagR + np.pi
            angHipSagL = angHipSagL + np.pi
            angKneR = np.pi - angKneR
            angKneL = np.pi - angKneL
            angAnkR = angAnkR + np.pi / 2.0
            angAnkL = angAnkL + np.pi / 2.0

            self.musHABR.stim = stimu[0]
            self.musHADR.stim = stimu[1]
            self.musGLUR.stim = stimu[2]
            self.musHFLR.stim = stimu[3]
            self.musHAMR.stim = stimu[4]
            self.musREFR.stim = stimu[5]
            self.musBFSHR.stim = stimu[6]
            self.musVASR.stim = stimu[7]
            self.musGASR.stim = stimu[8]
            self.musSOLR.stim = stimu[9]
            self.musTIAR.stim = stimu[10]
            self.musHABL.stim = stimu[11]
            self.musHADL.stim = stimu[12]
            self.musGLUL.stim = stimu[13]
            self.musHFLL.stim = stimu[14]
            self.musHAML.stim = stimu[15]
            self.musREFL.stim = stimu[16]
            self.musBFSHL.stim = stimu[17]
            self.musVASL.stim = stimu[18]
            self.musGASL.stim = stimu[19]
            self.musSOLL.stim = stimu[20]
            self.musTIAL.stim = stimu[21]
            self.musHABR.stepUpdateState(np.array((angHipAbdR,)))
            self.musHADR.stepUpdateState(np.array((angHipAbdR,)))
            self.musGLUR.stepUpdateState(np.array((angHipSagR,)))
            self.musHFLR.stepUpdateState(np.array((angHipSagR,)))
            self.musHAMR.stepUpdateState(np.array((angHipSagR, angKneR)))
            self.musREFR.stepUpdateState(np.array((angHipSagR, angKneR)))
            self.musBFSHR.stepUpdateState(np.array((angKneR,)))
            self.musVASR.stepUpdateState(np.array((angKneR,)))
            self.musGASR.stepUpdateState(np.array((angKneR, angAnkR)))
            self.musSOLR.stepUpdateState(np.array((angAnkR,)))
            self.musTIAR.stepUpdateState(np.array((angAnkR,)))
            self.musHABL.stepUpdateState(np.array((angHipAbdL,)))
            self.musHADL.stepUpdateState(np.array((angHipAbdL,)))
            self.musGLUL.stepUpdateState(np.array((angHipSagL,)))
            self.musHFLL.stepUpdateState(np.array((angHipSagL,)))
            self.musHAML.stepUpdateState(np.array((angHipSagL, angKneL)))
            self.musREFL.stepUpdateState(np.array((angHipSagL, angKneL)))
            self.musBFSHL.stepUpdateState(np.array((angKneL,)))
            self.musVASL.stepUpdateState(np.array((angKneL,)))
            self.musGASL.stepUpdateState(np.array((angKneL, angAnkL)))
            self.musSOLL.stepUpdateState(np.array((angAnkL,)))
            self.musTIAL.stepUpdateState(np.array((angAnkL,)))

            #print(self.musHABR.frcmtc, self.musHABL.frcmtc)
            torHipAbdR = self.musHABR.frcmtc * self.musHABR.levelArm - self.musHADR.frcmtc * self.musHADR.levelArm
            torHipExtR = self.musGLUR.frcmtc * self.musGLUR.levelArm - self.musHFLR.frcmtc * self.musHFLR.levelArm + \
                         self.musHAMR.frcmtc * self.musHAMR.levelArm[0] - self.musREFR.frcmtc * self.musREFR.levelArm[0]
            torKneFleR = self.musBFSHR.frcmtc * self.musBFSHR.levelArm - self.musVASR.frcmtc * self.musVASR.levelArm + \
                         self.musHAMR.frcmtc * self.musHAMR.levelArm[1] - self.musREFR.frcmtc * self.musREFR.levelArm[
                             1] + \
                         self.musGASR.frcmtc * self.musGASR.levelArm[0]
            torAnkExtR = self.musSOLR.frcmtc * self.musSOLR.levelArm - self.musTIAR.frcmtc * self.musTIAR.levelArm + \
                         self.musGASR.frcmtc * self.musGASR.levelArm[1]

            torHipAbdL = self.musHABL.frcmtc * self.musHABL.levelArm - self.musHADL.frcmtc * self.musHADL.levelArm
            torHipExtL = self.musGLUL.frcmtc * self.musGLUL.levelArm - self.musHFLL.frcmtc * self.musHFLL.levelArm + \
                         self.musHAML.frcmtc * self.musHAML.levelArm[0] - self.musREFL.frcmtc * self.musREFL.levelArm[0]
            torKneFleL = self.musBFSHL.frcmtc * self.musBFSHL.levelArm - self.musVASL.frcmtc * self.musVASL.levelArm + \
                         self.musHAML.frcmtc * self.musHAML.levelArm[1] - self.musREFL.frcmtc * self.musREFL.levelArm[
                             1] + \
                         self.musGASL.frcmtc * self.musGASL.levelArm[0]
            torAnkExtL = self.musSOLL.frcmtc * self.musSOLL.levelArm - self.musTIAL.frcmtc * self.musTIAL.levelArm + \
                         self.musGASL.frcmtc * self.musGASL.levelArm[1]
            tor = [-torHipAbdR, torHipExtR, torKneFleR, torAnkExtR,
                   torHipAbdL, torHipExtL, torKneFleL, torAnkExtL]
            # tor =  [stimu[0], stimu[1], stimu[2], stimu[3],
            #        stimu[4], stimu[5], stimu[6], stimu[7] ]
            self.sim.data.ctrl[:] = tor

            # run one step simulation
            self.sim.step()


    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.CoppeliaAgent.CoppeliaAgent import CoppeliaAgent
from crowd_sim.envs.CoppeliaAgent.lidar_data import lidar_data
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import threading
class Robot(Agent):
    init_flag = False
    def __init__(self, config, section,client):
        if Robot.init_flag:
            return
        super().__init__(config, section)
        self.set_client(client,name='Robot')
        self.map_list = np.zeros((10,16))
        self.action_map = np.zeros((10,16))
        self.last_map = np.zeros((10,16))
        Robot.init_flag = True
    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        self.last_map = np.zeros((10,16))
        state = JointState(self.get_full_state(), ob)
        robot_orientation = self.get_orientation()          # robot [-pi,pi]
        risk_map, diff_map, map = self.get_lidar_reading(self.last_map)
        action = self.policy.predict(state, risk_map, robot_orientation, map, diff_map)
        self.last_map = risk_map


        return action

    def action_risk_map(self,last_map,action):
        risk_map, diff_map, map = self.get_lidar_reading(last_map)
        state = JointState(self.get_full_state(), risk_map)
        self.action_map = lidar_data().action_range_danger(state.self_state.v_pref,diff_map,action,risk_map)
        return  self.action_map

    def set_client(self,client,name=None,id=None):
        self.agent = CoppeliaAgent(client,name,id)


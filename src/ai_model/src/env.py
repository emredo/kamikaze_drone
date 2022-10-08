import numpy as np
from msgs.msg import action, reset, state

class Environment:
    def __init__(self,action_publisher, reset_publisher,stopper_publisher):
        self.observation_space = np.array([8,1,1,1,1,1,1,1,1,1,1,1,1])
        self.action_space = np.array([8,1,1,1])
        self._action_publisher = action_publisher
        self._reset_publisher = reset_publisher
        self._stopper_publisher = stopper_publisher

    def step(self,action_array):
        msg = action()
        msg.ball1_vx = action_array[0]
        msg.ball1_vy = action_array[1]
        msg.ball1_vz = action_array[2]
        msg.ball1_angular = action_array[3]
        self._action_publisher.publish(msg)
    
    def observe(self,observation_msg):
        next_state = np.array([observation_msg.agent_x,
                                observation_msg.agent_y,
                                observation_msg.agent_z,
                                observation_msg.enemy_x,
                                observation_msg.enemy_y,
                                observation_msg.enemy_z,
                                observation_msg.agent_vx,
                                observation_msg.agent_vy,
                                observation_msg.agent_vz,
                                observation_msg.agent_w,
                                observation_msg.enemy_vx,
                                observation_msg.enemy_vy,
                                observation_msg.enemy_vz])
        rew = observation_msg.reward
        done = observation_msg.done_indice
        return next_state, rew, done

    def reset(self):
        msg = reset()
        msg.is_reset = True
        self._reset_publisher.publish(msg)
        # end_stop_msg = reset()
        # end_stop_msg.is_reset = False
        # self._stopper_publisher.publish(end_stop_msg)

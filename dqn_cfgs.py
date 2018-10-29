#!python3
__author__ = "Changjian Li"

import numpy as np
import tensorflow as tf

from include import *
from sumo_cfgs import *
from dqn import DQNCfg

def reshape_all(obs_dict):
  """reshape gym observation to keras neural network input"""
  # sqrt is used to strech the input to emphasize the near zero part
  o0 = np.array([np.sqrt(obs_dict["ego_speed"]/MAX_VEH_SPEED) - 0.5,
                 np.sqrt(min(obs_dict["ego_dist_to_end_of_lane"]/OBSERVATION_RADIUS, 1.0)) - 0.5,
                 obs_dict["ego_in_intersection"] - 0.5,
                 obs_dict["ego_exists_left_lane"] - 0.5,
                 obs_dict["ego_exists_right_lane"] - 0.5,
                 obs_dict["ego_correct_lane_gap"]/(2*NUM_LANE_CONSIDERED),
                 obs_dict["ego_has_priority"] - 0.5
                 ], dtype = np.float32)
  o1 = np.reshape(np.array([], dtype = np.float32), (0, NUM_VEH_CONSIDERED))
  o1  = np.append(o1, np.array([obs_dict["exists_vehicle"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["in_intersection"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["brake_signal"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["left_signal"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["right_signal"]]) - 0.5, axis=0)
  rel_speed = np.array([obs_dict["relative_speed"]]) / MAX_VEH_SPEED + 0.5
  rel_speed = np.minimum(np.sqrt(np.abs(rel_speed)), np.ones((1, NUM_VEH_CONSIDERED))*0.5) * np.sign(rel_speed)
  o1  = np.append(o1, rel_speed , axis=0)
  o1  = np.append(o1, np.sqrt(np.minimum(np.array([obs_dict["dist_to_end_of_lane"]])/OBSERVATION_RADIUS,
                              np.ones((1, NUM_VEH_CONSIDERED)))) - 0.5, axis = 0)
  rel_pos = np.array(obs_dict["relative_position"]).T / 2 * OBSERVATION_RADIUS
  rel_pos = np.sqrt(np.abs(rel_pos)) * np.sign(rel_pos)
  o1 = np.append(o1, rel_pos, axis=0)
  o1  = np.append(o1, np.array([obs_dict["relative_heading"]])/2*np.pi, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_peer"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_conflict"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_left"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_right"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_ahead"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_behind"]]) - 0.5, axis=0)
  ttc = np.array([obs_dict["ttc"]]) / MAX_TTC_CONSIDERED
  ttc = np.sqrt(np.abs(ttc)) * np.sign(ttc)
  o1 = np.append(o1, ttc - 0.5, axis=0)

  o = [o0] + [x for x in o1.T]
  return [[x] for x in o]

tf_cfg_all = tf.ConfigProto()
tf_cfg_all.gpu_options.per_process_gpu_memory_fraction = 0.8

def build_model_all():
  ego_input = tf.keras.layers.Input(shape=(7, ))
  ego_l1 = tf.keras.layers.Dense(64, activation=None)(ego_input)

  veh_inputs = [tf.keras.layers.Input(shape=(17,)) for _ in range(NUM_VEH_CONSIDERED)]
  veh_l = veh_inputs

  n_layers = 3
  Dense_list = [tf.keras.layers.Dense(64, activation=None) for _ in range(n_layers)]
  for i in range(n_layers):
    veh_l = [Dense_list[i](x) for x in veh_l]
    veh_l = [tf.keras.layers.Activation("sigmoid")(x) for x in veh_l]

  shared_Dense = tf.keras.layers.Dense(64, activation=None)
  veh_l = [shared_Dense(x) for x in veh_l]

  merged = tf.keras.layers.average(veh_l+[ego_l1])
  merged = tf.keras.layers.Activation("sigmoid")(merged)

  n_layers_merged = 1
  Dense_list_merged = [tf.keras.layers.Dense(64, activation=None) for _ in range(n_layers_merged)]
  for i in range(n_layers_merged):
    merged = Dense_list_merged[i](merged)
    merged = tf.keras.layers.Activation("sigmoid")(merged)

  y = tf.keras.layers.Dense(reduced_action_size, activation=None)(merged)

  model = tf.keras.models.Model(inputs=[ego_input] + veh_inputs, outputs=[y, y])
  opt = tf.keras.optimizers.RMSprop(lr=0.0001)
  model.compile(loss='logcosh', optimizer=opt)

  return model


class returnTrue():
  def __init__(self):
    pass
  def __call__(self, x):
    return True

cfg_all =    DQNCfg(name = "all",
                    play = False,
                    version = "current",
                    resume = False,
                    state_size = 5 + 17*NUM_VEH_CONSIDERED,
                    action_size = reduced_action_size,
                    low_target=-1,
                    high_target=0,
                    gamma = 0.9,
                    gamma_inc = 1e-5,
                    gamma_max = 0.9,
                    epsilon = 0.2,
                    epsilon_dec = 1e-5,
                    epsilon_min = 0.1,
                    threshold = -0.1,
                    memory_size = 3200,
                    traj_end_pred = returnTrue(),
                    replay_batch_size = 320,
                    traj_end_ratio= 0.0001,
                    _build_model = build_model_all,
                    model_rst_prob_list = [],
                    tf_cfg = tf_cfg_all,
                    reshape = reshape_all)
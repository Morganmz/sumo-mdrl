#!python3
__author__ = "Changjian Li"

import numpy as np
import tensorflow as tf

from include import *
from sumo_cfgs import *
from dqn import DQNCfg

def reshape_validity(obs_dict):
  out = np.array([obs_dict["ego_exists_left_lane"], obs_dict["ego_exists_right_lane"]], dtype=np.int32)
  return [np.reshape(out, (1, -1))]

def select_actions_validity(state):
    ego_exists_left_lane = state[0][0][0]
    ego_exists_right_lane = state[0][0][1]

    if ego_exists_left_lane == 0 and ego_exists_right_lane == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
               ]
      sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value]
    if ego_exists_left_lane == 0 and ego_exists_right_lane == 1:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
               ]
      sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value]
    if ego_exists_left_lane == 1 and ego_exists_right_lane == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
               ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
               ]
      sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value]
    if ego_exists_left_lane == 1 and ego_exists_right_lane == 1:
        valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                 ]
        sorted_idx = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                 ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value]

    return (set(valid), sorted_idx)

def reshape_safety(obs_dict):
  """reshape gym observation to keras neural network input"""
  o0 = np.array([np.sqrt(obs_dict["ego_speed"]/MAX_VEH_SPEED) - 0.5,
                 np.sqrt(min(obs_dict["ego_dist_to_end_of_lane"]/OBSERVATION_RADIUS, 1.0)) - 0.5,
                 obs_dict["ego_in_intersection"] - 0.5,
                 obs_dict["ego_exists_left_lane"] - 0.5,
                 obs_dict["ego_exists_right_lane"] - 0.5
                 ], dtype = np.float32)
  o1 = np.reshape(np.array([], dtype = np.float32), (0, NUM_VEH_CONSIDERED))
  o1  = np.append(o1, np.array([obs_dict["exists_vehicle"]]) - 0.5, axis=0)
  rel_speed = np.array([obs_dict["relative_speed"]]) / MAX_VEH_SPEED
  rel_speed = np.sqrt(np.minimum(np.abs(rel_speed), np.ones((1, NUM_VEH_CONSIDERED))*0.5)) * np.sign(rel_speed)
  o1  = np.append(o1, rel_speed , axis=0)
  o1  = np.append(o1, np.sqrt(np.minimum(np.array([obs_dict["dist_to_end_of_lane"]])/OBSERVATION_RADIUS,
                              np.ones((1, NUM_VEH_CONSIDERED)))) - 0.5, axis = 0)
  rel_pos = np.array(obs_dict["relative_position"]).T / 2 * OBSERVATION_RADIUS
  rel_pos = np.sqrt(np.abs(rel_pos)) * np.sign(rel_pos)
  o1 = np.append(o1, rel_pos, axis=0)
  o1  = np.append(o1, np.array([obs_dict["relative_heading"]])/2*np.pi, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_peer"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_conflict"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_next"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["veh_relation_prev"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_left"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_right"]]) - 0.5, axis=0)
  o1  = np.append(o1, np.array([obs_dict["veh_relation_ahead"]]) - 0.5, axis=0)
  o1 = np.append(o1, np.array([obs_dict["ttc"]]) / MAX_TTC_CONSIDERED - 0.5, axis=0)

  o = [o0] + [x for x in o1.T]
  return [[x] for x in o]

tf_cfg_safety = tf.ConfigProto()
tf_cfg_safety.gpu_options.per_process_gpu_memory_fraction = 0.4
#tf_cfg_safety = tf.ConfigProto(device_count = {"GPU": 0})

def build_model_safety():
  ego_input = tf.keras.layers.Input(shape=(5, ))
  ego_l1 = tf.keras.layers.Dense(640, activation=None)(ego_input)

  veh_inputs = [tf.keras.layers.Input(shape=(14,)) for _ in range(NUM_VEH_CONSIDERED)]
  shared_Dense1 = tf.keras.layers.Dense(640, activation=None)
  veh_l1 = [shared_Dense1(x) for x in veh_inputs]

  veh_l2 = [tf.keras.layers.add([ego_l1, x]) for x in veh_l1]
  veh_l2 = [tf.keras.layers.Activation('tanh')(x) for x in veh_l2]

  shared_Dense2 = tf.keras.layers.Dense(640, activation=None)
  veh_l3 = [shared_Dense2(x) for x in veh_l2]
  veh_l3 = [tf.keras.layers.Activation('tanh')(x) for x in veh_l3]

  shared_Dense3 = tf.keras.layers.Dense(640, activation=None)
  veh_l4 = [shared_Dense3(x) for x in veh_l3]
  veh_l4 = [tf.keras.layers.Activation('tanh')(x) for x in veh_l4]

  shared_Dense4 = tf.keras.layers.Dense(640, activation=None)
  veh_l5 = [shared_Dense4(x) for x in veh_l4]
  veh_l5 = [tf.keras.layers.Activation('tanh')(x) for x in veh_l5]

  shared_Dense5 = tf.keras.layers.Dense(640, activation=None)
  veh_l6 = [shared_Dense5(x) for x in veh_l5]
  veh_l6 = [tf.keras.layers.Activation('tanh')(x) for x in veh_l6]

  shared_Dense6 = tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation=None)
  veh_y = [shared_Dense6(x) for x in veh_l6]
  y = tf.keras.layers.add(veh_y)

  model = tf.keras.models.Model(inputs=[ego_input] + veh_inputs, outputs=veh_y + [y])
  opt = tf.keras.optimizers.SGD(lr=0.001)
  model.compile(loss='logcosh', optimizer=opt)

  return model

def reshape_regulation(obs_dict):
  lane_gap_1hot = [-0.5] * (2*NUM_LANE_CONSIDERED + 1)
  lane_gap_1hot[obs_dict["ego_correct_lane_gap"] + NUM_LANE_CONSIDERED] = 0.5

  o = np.array([np.sqrt(obs_dict["ego_speed"]/MAX_VEH_SPEED) - 0.5,
                np.sqrt(min(obs_dict["ego_dist_to_end_of_lane"] / OBSERVATION_RADIUS, 1.0)) - 0.5,
                obs_dict["ego_in_intersection"] - 0.5,
                obs_dict["ego_has_priority"] - 0.5,
                ] + lane_gap_1hot, dtype = np.float32)

  return [[o]]

tf_cfg_regulation = tf.ConfigProto()
tf_cfg_regulation.gpu_options.per_process_gpu_memory_fraction = 0.4

def build_model_regulation():
  x = tf.keras.layers.Input(shape=(5 + 2*NUM_LANE_CONSIDERED, ))
  l1 = tf.keras.layers.Dense(640, activation=None)(x)
  l1 = tf.keras.layers.Activation('tanh')(l1)
  l2 = tf.keras.layers.Dense(640, activation=None)(l1)
  l2 = tf.keras.layers.Activation('tanh')(l2)
  l3 = tf.keras.layers.Dense(640, activation=None)(l2)
  l3 = tf.keras.layers.Activation('tanh')(l3)
  l4 = tf.keras.layers.Dense(640, activation=None)(l3)
  l4 = tf.keras.layers.Activation('tanh')(l4)
  y = tf.keras.layers.Dense(len(ActionLaneChange) * len(ActionAccel), activation='linear')(l4)

  model = tf.keras.models.Model(inputs=[x], outputs=[y, y])
  opt = tf.keras.optimizers.SGD(lr=0.001)
  model.compile(loss='logcosh', optimizer=opt)
  return model

def reshape_speed_comfort(obs_dict):
  out = np.array([obs_dict["ego_speed"], obs_dict["ego_correct_lane_gap"]], dtype = np.float32)
  return [np.reshape(out, (1,-1))]

def select_actions_speed_comfort(state):
  ego_speed = state[0][0][0]
  ego_correct_lane_gap = state[0][0][1]
  if ego_speed < MAX_VEH_SPEED-1.4:
    if ego_correct_lane_gap == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value]
      sorted_idx  = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    elif ego_correct_lane_gap > 0:
      valid = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
               ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value]
      sorted_idx  = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    else:
      valid = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value]
      sorted_idx  = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
  elif ego_speed > MAX_VEH_SPEED + 1.4:
    if ego_correct_lane_gap == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value
                     ]
    elif ego_correct_lane_gap > 0:
      valid = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    else:
      valid = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
  else:
    if ego_correct_lane_gap == 0:
      valid = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    elif ego_correct_lane_gap > 0:
      valid = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
    elif ego_correct_lane_gap < 0:
      valid = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value]
      sorted_idx  = [ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.NOOP.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.RIGHT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.NOOP.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MINDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXACCEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MEDDECEL.value,
                     ActionLaneChange.LEFT.value * len(ActionAccel) + ActionAccel.MAXDECEL.value
                     ]
  return (set(valid), sorted_idx)

action_size = len(ActionLaneChange) * len(ActionAccel)

cfg_validity = DQNCfg(name = "validity",
                      play=False,
                      resume = False,
                      state_size=2,
                      action_size=action_size,
                      low_target=None,
                      high_target=None,
                      gamma=None,
                      gamma_inc=None,
                      gamma_max=None,
                      epsilon=0,
                      epsilon_dec=0,
                      epsilon_min=0,
                      threshold=None,
                      memory_size=None,
                      traj_end_pred=None,
                      replay_batch_size=None,
                      traj_end_ratio= None,
                      _build_model=None,
                      tf_cfg=None,
                      reshape=reshape_validity,
                      _select_actions=select_actions_validity)

class returnTrue():
  def __init__(self):
    pass
  def __call__(self, x):
    return True

cfg_safety = DQNCfg(name = "safety",
                    play = False,
                    resume = False,
                    state_size = 5 + 12*NUM_VEH_CONSIDERED,
                    action_size = action_size,
                    low_target=-1,
                    high_target=0,
                    gamma = 0.9,
                    gamma_inc = 0.00000001,
                    gamma_max = 0.95,
                    epsilon = 0.8,
                    epsilon_dec = 0.0000001,
                    epsilon_min = 0.6,
                    threshold = -0.05,
                    memory_size = 64000,
                    traj_end_pred = returnTrue(),
                    replay_batch_size = 32,
                    traj_end_ratio= 0.0001,
                    _build_model = build_model_safety,
                    tf_cfg = tf_cfg_safety,
                    reshape = reshape_safety)

cfg_regulation = DQNCfg(name = "regulation",
                        play = False,
                        resume = False,
                        state_size = 4 + 2*NUM_LANE_CONSIDERED + 7*NUM_VEH_CONSIDERED,
                        action_size = action_size,
                        low_target=-10,
                        high_target=0,
                        gamma = 0.90,
                        gamma_inc = 0.00000001,
                        gamma_max = 0.95,
                        epsilon=0.8,
                        epsilon_dec=0.0000001,
                        epsilon_min=0.6,
                        threshold = -5,
                        memory_size = 64000,
                        traj_end_pred = returnTrue(),
                        replay_batch_size = 160,
                        traj_end_ratio= 0.0001,
                        _build_model = build_model_regulation,
                        tf_cfg = tf_cfg_regulation,
                        reshape = reshape_regulation)

cfg_speed_comfort = DQNCfg(name = "speed_comfort",
                           play = False,
                           resume=False,
                           state_size = 2,
                           action_size = action_size,
                           low_target=None,
                           high_target=None,
                           gamma = None,
                           gamma_inc = None,
                           gamma_max = None,
                           epsilon=0,
                           epsilon_dec=0,
                           epsilon_min=0,
                           threshold = None,
                           memory_size = None,
                           traj_end_pred = None,
                           replay_batch_size = None,
                           traj_end_ratio= None,
                           _build_model = None,
                           tf_cfg = None,
                           reshape = reshape_speed_comfort,
                           _select_actions=select_actions_speed_comfort)
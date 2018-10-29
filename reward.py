#!/bin/python3
__author__ = "Changjian Li"

from include import *
import numpy as np

def get_reward_list(env):
  r, d, violation_safety, violation_yield, violation_turn = get_reward_all(env)

  return ([r],
          [d],
          [violation_safety, violation_yield, violation_turn])

def get_reward_all(env):
  obs_dict = env.obs_dict_hist[-1]
  old_obs_dict = None
  if len(env.obs_dict_hist) > 1:
    old_obs_dict = env.obs_dict_hist[-2]
  action_dict = env.action_dict_hist[-1]

  violated_safety = False
  d = False
  if env.env_state == EnvState.CRASH:
    violated_safety = True
    d = True

  r = 0
  for i, c in enumerate(obs_dict["collision"]):

    if (old_obs_dict is not None and
        obs_dict["is_new"][i] == 0 and
        obs_dict["veh_relation_none"] != 1 and
        (obs_dict["veh_relation_ahead"][i] == 1 or
         ((obs_dict["veh_relation_conflict"][i] == 1 or obs_dict["veh_relation_peer"][i] == 1) and
          obs_dict["in_intersection"][i] == 1 and
          obs_dict["ego_in_intersection"] == 1)
         ) and
        ((abs(old_obs_dict["ttc"][i]) > abs(obs_dict["ttc"][i]) + 1e-6 and
          (np.linalg.norm(old_obs_dict["relative_position"][i]) < 8 or old_obs_dict["ttc"][i] < 4) and
          (action_dict["accel_level"] != ActionAccel.MAXDECEL or obs_dict["veh_relation_behind"][i] == 1)) or
         (np.linalg.norm(old_obs_dict["relative_position"][i]) > np.linalg.norm(obs_dict["relative_position"][i]) + 1e-6 and
          np.linalg.norm(old_obs_dict["relative_position"][i]) < 7
         ))
        ) or (env.env_state == EnvState.CRASH and c == 1
        ) or (action_dict["lane_change"] != ActionLaneChange.NOOP and (obs_dict["ttc"][i] < 1)
        ):
      print(obs_dict["veh_ids"][i], "old_ttc", old_obs_dict["ttc"][i], "ttc", obs_dict["ttc"][i],
            "pos", np.linalg.norm(old_obs_dict["relative_position"][i]), "action", action_dict,
            "collision", c)
      r = -1
    if r == -1:
      d = True

  violated_turn = False
  violated_yield = False

  if obs_dict["ego_correct_lane_gap"] != 0:
    r += 0.3 * (1 / (1 + np.exp(-0.1 * (obs_dict["ego_dist_to_end_of_lane"] - 60))) - 1)

  old_tte = None
  if old_obs_dict is not None:
    old_tte = old_obs_dict["ego_dist_to_end_of_lane"] / (old_obs_dict["ego_speed"] + 1e-6)
  tte = obs_dict["ego_dist_to_end_of_lane"] / (obs_dict["ego_speed"] + 1e-6)
  if old_tte is not None and \
     (old_tte < 4 or old_obs_dict["ego_dist_to_end_of_lane"] < 1) and \
     obs_dict["ego_has_priority"] != 1 and \
     obs_dict["ego_in_intersection"] != 1 and \
     old_tte > tte + 1e-6:
      print("regulation: old_tte",old_tte , " tte ", tte)
      r += -1

  if obs_dict["ego_speed"] < 1:
    r += -0.1

  if r <= -1:
    r = -1
    d = True

  if old_tte is not None and \
     obs_dict["ego_has_priority"] == 1 and \
     obs_dict["ego_in_intersection"] != 1 and \
     old_tte < tte - 1e-6:
      r = -0.02

  if (tte < 0.15 and obs_dict["ego_correct_lane_gap"] != 0):
    violated_turn = True
  if (old_obs_dict is not None and old_obs_dict["ego_has_priority"] != 1 and
      old_obs_dict["ego_in_intersection"] != 1 and obs_dict["ego_in_intersection"] == 1):
    violated_yield = True


  return ([[r]], [[d]], violated_safety, violated_yield, violated_turn)

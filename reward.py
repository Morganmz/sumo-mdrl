#!/bin/python3
__author__ = "Changjian Li"

from include import *
import numpy as np

def get_reward_list(env):
  r, d, violation0, violation1 = get_reward_all(env)

  return ([r],
          [d],
          [violation0, violation1])

def get_reward_all(env):
  obs_dict = env.obs_dict_hist[-1]
  old_obs_dict = None
  if len(env.obs_dict_hist) > 1:
    old_obs_dict = env.obs_dict_hist[-2]
  action_dict = env.action_dict_hist[-1]

  violated0 = False
  d = False
  if env.env_state == EnvState.CRASH:
    violated0 = True
    d = True

  r = 0
  for i, c in enumerate(obs_dict["collision"]):

    if (old_obs_dict is not None and
        obs_dict["is_new"][i] == 0 and
        obs_dict["veh_relation_none"] != 1 and
        (obs_dict["veh_relation_ahead"][i] == 1 or
         obs_dict["veh_relation_behind"][i] == 1 or
         ((obs_dict["veh_relation_conflict"][i] == 1 or obs_dict["veh_relation_peer"][i] == 1) and
          (obs_dict["ego_has_priority"] == 0 or obs_dict["in_intersection"][i] == 1) and
          old_obs_dict["dist_to_end_of_lane"][i] < 20 and
          old_obs_dict["ego_dist_to_end_of_lane"] < 20 and
          (obs_dict["ego_in_intersection"] != 1 or (obs_dict["ego_in_intersection"] == 1 and obs_dict["in_intersection"][i] == 1)))
         ) and
        (abs(old_obs_dict["ttc"][i]) > abs(obs_dict["ttc"][i]) + 0.0000001 and
         (np.linalg.norm(old_obs_dict["relative_position"][i]) < 8 or old_obs_dict["ttc"][i] < 3) and
         action_dict["accel_level"] != ActionAccel.MAXDECEL
         )
        ) or (env.env_state == EnvState.CRASH
        ) or (action_dict["lane_change"] != ActionLaneChange.NOOP and (obs_dict["ttc"][i] < 2)
        ):
      print(obs_dict["veh_ids"][i], "old_ttc", old_obs_dict["ttc"][i], "ttc", obs_dict["ttc"][i],
            "pos", np.linalg.norm(old_obs_dict["relative_position"][i]), "action", action_dict,
            "collision", c)
      r = -1

  # regulation
  violated1 = False
  if (obs_dict["ego_dist_to_end_of_lane"] < 0.01 and obs_dict["ego_correct_lane_gap"] != 0
  ) or (tte < 0.5 and obs_dict["ego_has_priority"] != 1 and obs_dict["ego_in_intersection"] != 1):
    violated1 = True

  if obs_dict["ego_correct_lane_gap"] != 0:
    r += 0.5 * (1 / (1 + np.exp(-0.1 * (obs_dict["ego_dist_to_end_of_lane"] - 60))) - 1)

  old_tte = None
  if old_obs_dict is not None:
    old_tte = old_obs_dict["ego_dist_to_end_of_lane"] / (old_obs_dict["ego_speed"] + 0.00000001)
  tte = obs_dict["ego_dist_to_end_of_lane"] / (obs_dict["ego_speed"] + 0.00000001)
  if ((old_tte is not None and old_tte < 3) or old_obs_dict["ego_dist_to_end_of_lane"] < 2) and \
     obs_dict["ego_has_priority"] != 1 and \
     obs_dict["ego_in_intersection"] != 1 and \
     old_tte > tte + 0.00001 and \
     action_dict["accel_level"] != ActionAccel.MAXDECEL:
      print("regulation: old_tte",old_tte , " tte ", tte)
      r += -1

  return ([[r]], [[d]], violated0, violated1)


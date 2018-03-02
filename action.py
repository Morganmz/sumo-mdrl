#!python3
__author__ = "Changjian Li, Aman Jhunjhunwala"

from include import *

def get_action_space():
  action_space = spaces.Dict({"lane_change": spaces.Discrete(len(ActionLaneChange)),
                            "accel_level": spaces.Discrete(len(ActionAccel))
                           })
  return action_space
  
def disable_collision_check(env, veh_id):
  print("disabled")
  env.tc.vehicle.setSpeedMode(veh_id, 0b00000)
  env.tc.vehicle.setLaneChangeMode(veh_id, 0b0000000000)
  
def enable_collision_check(env, veh_id):
  print("enabled")
  env.tc.vehicle.setSpeedMode(veh_id, 0b11111)
  env.tc.vehicle.setLaneChangeMode(veh_id, 0b011001010101)

def is_illegal_action(env, veh_id, action_dict):
  """ illegal action is an action that will lead to problems such as a env.tc exception
  """
  # couldChangeLane has a time lag of one step, a workaround is needed until this is fixed
  #if (action_dict["lane_change"] == 1 and env.tc.vehicle.couldChangeLane(veh_id, 1) == False) or \
     #(action_dict["lane_change"] == 2 and env.tc.vehicle.couldChangeLane(veh_id, -1) == False):
  num_lanes_veh_edge = env.tc.edge.getLaneNumber(env.tc.vehicle.getRoadID(veh_id))
  if (action_dict["lane_change"] == ActionLaneChange.LEFT and env.tc.vehicle.getLaneIndex(veh_id) == num_lanes_veh_edge - 1) or \
     (action_dict["lane_change"] == ActionLaneChange.RIGHT and env.tc.vehicle.getLaneIndex(veh_id) == 0):
    return True
  return False 

def is_invalid_action(env, veh_id, action_dict):
  """ invalid action is an action that doesn't make sense, it's treated as a noop
  """
  return False

def inc_speed(speed, inc, max_speed):
  #print("speed inc: ", inc)
  if (speed + inc) > max_speed:
    return max_speed
  else:
    return speed + inc

def dec_speed(speed, dec, min_speed):
  #print("speed dec: ", dec)
  if (speed - dec) < min_speed:
    return min_speed
  else:
    return speed - dec

def act(env, veh_id, action_dict):
  """ take one simulation step with vehicles acting according to veh_id_and_action_list = [(veh_id0, action0), (veh_id1, action1), ...], 
      return True if an invalid action is taken or any of the vehicles collide.
  """
  if veh_id not in env.tc.vehicle.getIDList():
    return EnvState.DONE
    
  # An illegal action is considered as causing a collision
  if is_illegal_action(env, veh_id, action_dict):
    return EnvState.CRASH
    
  # action set to noop if it's invalid
  if is_invalid_action(env, veh_id, action_dict):
    action_dict = {"lane_change": ActionLaneChange.NOOP, "accel_level": ActionAccel.NOOP}
      
  # if car is controlled by RL agent
  if env.agt_ctrl == True:
    
    #print("entering setSpeed")
    # Lane Change
    if action_dict["lane_change"] == ActionLaneChange.LEFT:
      env.tc.vehicle.changeLane(veh_id, env.tc.vehicle.getLaneIndex(veh_id) + 1, int(env.SUMO_TIME_STEP * 1000)-1)
    elif action_dict["lane_change"] == ActionLaneChange.RIGHT:
      env.tc.vehicle.changeLane(veh_id, env.tc.vehicle.getLaneIndex(veh_id) - 1, int(env.SUMO_TIME_STEP * 1000)-1)
    else:
      pass
  
    ego_speed = env.tc.vehicle.getSpeed(veh_id)
    ego_max_speed = min(env.tc.vehicle.getAllowedSpeed(veh_id), env.MAX_VEH_SPEED)
    ego_max_accel = min(env.tc.vehicle.getAccel(veh_id), env.MAX_VEH_ACCEL)
    ego_max_decel = min(env.tc.vehicle.getDecel(veh_id), env.MAX_VEH_DECEL)

    # Accelerate/Decelerate
    accel_level = action_dict["accel_level"]
    if accel_level.value > ActionAccel.NOOP.value:
      ego_next_speed = inc_speed(ego_speed, (accel_level.value - ActionAccel.NOOP.value)/(ActionAccel.MAXACCEL.value - ActionAccel.NOOP.value) * ego_max_accel * env.SUMO_TIME_STEP, ego_max_speed)
    elif accel_level.value < ActionAccel.NOOP.value:
      ego_next_speed = dec_speed(ego_speed, (-accel_level.value + ActionAccel.NOOP.value)/(-ActionAccel.MAXDECEL.value + ActionAccel.NOOP.value) * ego_max_decel * env.SUMO_TIME_STEP, 0)
    else:
      # if car is controlled by RL agent, then ActionAccel.NOOP maintains the current speed
      ego_next_speed = ego_speed
    #env.tc.vehicle.slowDown(veh_id, ego_next_speed, int(env.SUMO_TIME_STEP * 1000)-1)
    env.tc.vehicle.setSpeed(veh_id, ego_next_speed)

    # Turn not implemented

  env.tc.simulationStep()
  
  if env.tc.simulation.getCollidingVehiclesNumber() > 0:
    if veh_id in env.tc.simulation.getCollidingVehiclesIDList():
      return EnvState.CRASH
  # if the subject vehicle goes out of scene, set env.env_state to EnvState.DONE
  if veh_id not in env.tc.vehicle.getIDList():
    return EnvState.DONE
  
  return EnvState.NORMAL


def infer_action(env):
  """When ego vehicle is controlled by sumo, the action taken in that time step need to be inferred
  """
  veh_dict = env.veh_dict_hist.get(-2)
  new_veh_dict = env.veh_dict_hist.get(-1)
  if new_veh_dict[env.EGO_VEH_ID]["edge_id"] == veh_dict[env.EGO_VEH_ID]["edge_id"]:
    if new_veh_dict[env.EGO_VEH_ID]["lane_index"] - veh_dict[env.EGO_VEH_ID]["lane_index"] == 1:
      lane_change = ActionLaneChange.LEFT.value 
    elif new_veh_dict[env.EGO_VEH_ID]["lane_index"] - veh_dict[env.EGO_VEH_ID]["lane_index"] == -1:
      lane_change = ActionLaneChange.RIGHT.value
    else:
      lane_change = ActionLaneChange.NOOP.value
  else:
    lane_change = ActionLaneChange.NOOP.value
  ego_max_accel = min(env.tc.vehicle.getAccel(env.EGO_VEH_ID), env.MAX_VEH_ACCEL)
  ego_max_decel = min(env.tc.vehicle.getDecel(env.EGO_VEH_ID), env.MAX_VEH_DECEL)
  
  #print("max veh accel: ", env.tc.vehicle.getAccel(env.EGO_VEH_ID), "max accel capability: ", env.MAX_VEH_ACCEL)
  #print("max veh decel: ", env.tc.vehicle.getDecel(env.EGO_VEH_ID), "max decel capability: ", env.MAX_VEH_DECEL)
  #print("max veh speed: ", env.tc.vehicle.getAllowedSpeed(env.EGO_VEH_ID), "max speed capability: ", env.MAX_VEH_SPEED)
  accel = (new_veh_dict[env.EGO_VEH_ID]["speed"] - veh_dict[env.EGO_VEH_ID]["speed"])/env.SUMO_TIME_STEP
  if accel >= 0:
    if accel/ego_max_accel <= 1/6:
      accel_level = ActionAccel.NOOP.value
    elif accel/ego_max_accel > 1/6 and accel/ego_max_accel <= 1/2:
      accel_level = ActionAccel.MINACCEL.value
    elif accel/ego_max_accel > 1/2 and accel/ego_max_accel <= 5/6:
      accel_level = ActionAccel.MEDACCEL.value
    else:
      accel_level = ActionAccel.MAXACCEL.value
  if accel < 0:
    accel = -accel
    if accel/ego_max_decel <= 1/6:
      accel_level = ActionAccel.NOOP.value
    elif accel/ego_max_accel > 1/6 and accel/ego_max_accel <= 1/2:
      accel_level = ActionAccel.MINDECEL.value
    elif accel/ego_max_accel > 1/2 and accel/ego_max_accel <= 5/6:
      accel_level = ActionAccel.MEDDECEL.value
    else:
      accel_level = ActionAccel.MAXDECEL.value
  action_dict = {"lane_change":ActionLaneChange(lane_change), "accel_level":ActionAccel(accel_level)}
  return action_dict

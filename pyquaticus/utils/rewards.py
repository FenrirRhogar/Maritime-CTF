# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-ClauseERROR: Package 'pyquaticus' requires a different Python: 3.13.13 not in 

"""
#Configureable Rewards
    # -- NOTE --
    #   All headings are in nautical format
    #                 0
    #                 |
    #          270 -- . -- 90
    #                 |
    #                180
    #
    # This can be converted the standard heading format that is counterclockwise
    # by using the heading_angle_conversion(deg) function found in utils.py
    #
    #
    ## Each custom reward function should have the following arguments ##
    Args:
        agent_id (int): ID of the agent we are computing the reward for
        team (Team): team of the agent we are computing the reward for
        agents (list): list of agent ID's (this is used to map agent_id's to agent indices and viceversa)
        agent_inds_of_team (dict): mapping from team to agent indices of that team
        state (dict):
            'agent_position' (array): list of agent positions (indexed in the order of agents list)

                        Ex. Usage: Get agent's current position
                        agent_id = 'agent_1'
                        position = state['agent_position'][agents.index(agent_id)]

            'prev_agent_position' (array): list of agent positions (indexed in the order of agents list) at the previous timestep

                        Ex. Usage: Get agent's previous position
                        agent_id = 'agent_1'
                        prev_position = state['prev_agent_position'][agents.index(agent_id)]

            'agent_speed' (array): list of agent speeds (indexed in the order of agents list)

                        Ex. Usage: Get agent's speed
                        agent_id = 'agent_1'
                        speed = state

            'agent_heading' (array): list of agent headings (indexed in the order of agents list)

                        Ex. Usage: Get agent's heading
                        agent_id = 'agent_1'
                        heading = state['agent_heading'][agents.index(agent_id)]

            'agent_on_sides' (array): list of booleans (indexed in the order of agents list) where True means the
                                      agent is on its own side, and False means the agent is not on its own side

                        Ex. Usage: Check if agent is on its own side
                        agent_id = 'agent_1'
                        on_own_side = state['agent_on_sides'][agents.index(agent_id)]

            'agent_oob' (array): list of booleans (indexed in the order of agents list) where True means the
                                 agent is out-of-bounds (OOB), and False means the agent is not out-of-bounds
                        
                        Ex. Usage: Check if agent is out-of-bounds
                        agent_id = 'agent_1'
                        num_oob = state['agent_oob'][agents.index(agent_id)]
            
            'agent_has_flag' (array): list of booleans (indexed in the order of agents list) where True means the
                                     agent has a flag, and False means the agent does not have a flag

                        Ex. Usage: Check if agent has a flag
                        agent_id = 'agent_1'
                        has_flag = state['agent_has_flag'][agents.index(agent_id)]

            'agent_is_tagged' (array): list of booleans (indexed in the order of agents list) where True means
                                       the agent is tagged, and False means the agent is not tagged

                        Ex. Usage: Check if agent is tagged
                        agent_id = 'agent_1'
                        is_tagged = state['agent_is_tagged'][agents.index(agent_id)]

            'agent_made_tag' (array): list (indexed in the order of agents list) where the value at an entry is the index of a different
                                     agent which the agent at the given index has tagged at the current timestep, otherwise None

                        Ex. Usage: Check if agent has tagged an agent
                        agent_id = 'agent_1'
                        tagged_opponent_idx = state['agent_made_tag'][agents.index(agent_id)]

            'agent_tagging_cooldown' (array): current agent tagging cooldowns (indexed in the order of agents list)
                        Note: agent is able to tag when this value is equal to tagging_cooldown
    
                        Ex. Usage: Get agent's current tagging cooldown
                        agent_id = 'agent_1'
                        cooldown = self.state['agent_tagging_cooldown'][agents.index(agent_id)]

            'dist_bearing_to_obstacles' (dict): For each agent in game list out distances and bearings
                                                to all obstacles in game in order of obstacles list

            'flag_home' (array): list of flag homes (indexed by team number)

            'flag_position' (array): list of flag homes (indexed by team number)

            'flag_taken' (array): list of booleans (indexed by team number) where True means the team's flag
                                  is taken (picked up by an opponent), and False means the flag is not taken 

            'team_has_flag' (array): list of booleans (indexed by team number) where True means an agent of the
                                     team has a flag, and False means that no agents are in possesion of a flag

            'captures' (array): list of total captures made by each team (indexed by team number)

            'tags' (array): list of total tags made by each team (indexed by team number)

            'grabs' (array): list of total flag grabs made by each team (indexed by team number)

            'agent_collisions' (array): list of total agent collisions  for each agent (indexed in the order of agents list)

            'agent_dynamics' (array): list of dictionaries containing agent-specific dynamics information (state attribute of a dynamics class - see dynamics.py)

            ######################################################################################
            ##### The following keys will exist in the state dictionary if lidar_obs is True #####
                'lidar_labels' (dict):

                'lidar_labels' (dict):

                'lidar_labels' (dict):
            ######################################################################################
            
            'obs_hist_buffer' (dict): Observation history buffer where the keys are agent_id's and values are the agents' observations

            'global_state_hist_buffer' (array): Global state history buffer

        prev_state (dict): Contains the state information from the previous step

        env_size (array): field dimensions [horizontal, vertical]

        agent_radii (array): list of agent radii (indexed in the order of agents list)

        catch_radius (float): tag and flag grab radius

        scrimmage_coords (array): endpoints [x,y] of the scrimmage line

        max_speeds (list): list of agent max speeds (indexed in the order of agents list)

        tagging_cooldown (float): tagging cooldown time
"""

import math
import numpy

from pyquaticus.structs import Team
from pyquaticus.utils.utils import *

### Example Reward Funtion ###
def example_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
):
    return 0.0

def caps_and_grabs(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
    ):
    reward = 0.0
    prev_num_oob = prev_state['agent_oob'][agents.index(agent_id)]
    num_oob = state['agent_oob'][agents.index(agent_id)]
    if num_oob > prev_num_oob:
        reward += -1.0

    #Check if agents lost flag
    prev_has_flag = prev_state['agent_has_flag'][agents.index(agent_id)]
    has_flag = state['agent_has_flag'][agents.index(agent_id)]
    #Agent lost flag
    if (prev_has_flag > has_flag): 
        reward += -0.25
    
    #Grabs and captures are of shape [team_0 (BLUE), team_1 (RED)] the value at the index 0 corresponds to the number of grabs
    for t in range(len(state['grabs'])):
        prev_num_grabs = prev_state['grabs'][t]
        num_grabs = state['grabs'][t]
        if num_grabs > prev_num_grabs:
            reward += 0.25 if t == int(team) else -0.25

        prev_num_caps = prev_state['captures'][t]
        num_caps = state['captures'][t]
        if num_caps > prev_num_caps:
            reward += 1.0 if t == int(team) else -1.0

    return reward

### Add Custom Reward Functions Here ###

# --- Helper Functions for Custom Dense Reward ---

def _reward_border_penalty(state, agent_idx, env_size, buffer_dist, oob_penalty, proximity_max_penalty):
    """Calculates penalties for being near or out of boundaries."""
    reward = 0.0
    pos = state['agent_position'][agent_idx]
    x, y = pos[0], pos[1]
    width, height = env_size[0], env_size[1]
    
    # Large penalty for going Out-Of-Bounds (OOB)
    if state['agent_oob'][agent_idx]:
        reward -= oob_penalty
    
    # Small progressive penalty for being too close to the edge
    dist_to_left = x
    dist_to_right = width - x
    dist_to_bottom = y
    dist_to_top = height - y
    min_dist_to_edge = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
    
    if min_dist_to_edge < buffer_dist:
        proximity_penalty = proximity_max_penalty * (1.0 - (min_dist_to_edge / buffer_dist))
        reward -= proximity_penalty
    return reward

def _reward_goal_progress(state, prev_state, agent_idx, team, progress_multiplier):
    """Rewards moving closer to the opponent's flag when on their side."""
    reward = 0.0
    
    # Only reward if we DON'T have the flag yet (we are hunting it)
    if state['agent_has_flag'][agent_idx]:
        return 0.0

    # Only reward if we are on the opponent's side
    if state['agent_on_sides'][agent_idx]:
        return 0.0

    opponent_team = 1 - int(team) # If we are 0 (Blue), opponent is 1 (Red)
    flag_pos = state['flag_position'][opponent_team]
    
    curr_pos = state['agent_position'][agent_idx]
    prev_pos = prev_state['agent_position'][agent_idx]
    
    dist_curr = np.linalg.norm(curr_pos - flag_pos)
    dist_prev = np.linalg.norm(prev_pos - flag_pos)
    
    # If distance decreased, we made progress
    if dist_curr < dist_prev:
        # Reward is proportional to how close we are (higher reward when very close)
        progress = dist_prev - dist_curr
        closeness_multiplier = 100.0 / max(dist_curr, 1.0) 
        reward += progress * closeness_multiplier * progress_multiplier
        
    return reward

def _reward_flag_grab(state, prev_state, agent_idx, grab_reward):
    """Provides a one-time reward when the agent successfully grabs the flag."""
    # Check if the agent just acquired the flag in this step
    if state['agent_has_flag'][agent_idx] and not prev_state['agent_has_flag'][agent_idx]:
        return grab_reward
    return 0.0

def _reward_return_progress(state, prev_state, agent_idx, team, progress_multiplier):
    """Rewards moving closer to home base when carrying the flag."""
    reward = 0.0
    
    # Only reward if we HAVE the flag
    if not state['agent_has_flag'][agent_idx]:
        return 0.0

    home_pos = state['flag_home'][int(team)]
    
    curr_pos = state['agent_position'][agent_idx]
    prev_pos = prev_state['agent_position'][agent_idx]
    
    dist_curr = np.linalg.norm(curr_pos - home_pos)
    dist_prev = np.linalg.norm(prev_pos - home_pos)
    
    # If distance decreased, we made progress
    if dist_curr < dist_prev:
        progress = dist_prev - dist_curr
        closeness_multiplier = 100.0 / max(dist_curr, 1.0) 
        reward += progress * closeness_multiplier * progress_multiplier
        
    return reward

def _reward_flag_capture(state, prev_state, team, capture_reward):
    """Provides a reward when the team successfully captures a flag."""
    team_idx = int(team)
    if state['captures'][team_idx] > prev_state['captures'][team_idx]:
        return capture_reward
    return 0.0

def _reward_tagging(state, agent_idx, team, base_tag_reward, flag_tag_reward):
    """Rewards tagging opponents, with a bonus for tagging the flag carrier."""
    reward = 0.0
    tagged_opponent_idx = state['agent_made_tag'][agent_idx]
    
    if tagged_opponent_idx is not None:
        # 1. Base reward for tagging any opponent
        reward += base_tag_reward
        
        # 2. Bonus reward if the tagged opponent was carrying our flag
        # (This makes the agent a "Hero" defender)
        if state['agent_has_flag'][tagged_opponent_idx]:
            reward += flag_tag_reward
            
    return reward

def _reward_teammate_proximity(state, agent_idx, team, agent_inds_of_team, avoid_dist, max_penalty):
    """Penalizes being too close to teammates to encourage spreading out."""
    reward = 0.0
    my_pos = state['agent_position'][agent_idx]
    
    # Get indices of all teammates
    teammate_indices = agent_inds_of_team[team]
    
    for tm_idx in teammate_indices:
        if tm_idx == agent_idx:
            continue  # Don't compare to self
            
        tm_pos = state['agent_position'][tm_idx]
        dist = np.linalg.norm(my_pos - tm_pos)
        
        if dist < avoid_dist:
            # Penalty scales from 0.0 (at avoid_dist) to max_penalty (at 0m)
            proximity_factor = 1.0 - (dist / avoid_dist)
            reward -= proximity_factor * max_penalty
            
    return reward

def _reward_tagged_penalty(state, prev_state, agent_idx, penalty):
    """Penalizes the agent for being tagged by an opponent."""
    if state['agent_is_tagged'][agent_idx] and not prev_state['agent_is_tagged'][agent_idx]:
        return -penalty
    return 0.0

def _reward_intercept_progress(state, prev_state, agent_idx, team, agents, agent_inds_of_team, progress_multiplier):
    """Rewards moving closer to the opponent who has our flag."""
    reward = 0.0
    
    # Find if any opponent has our flag
    # Opponents are agents NOT in our team's indices
    opponent_carrier_idx = None
    my_teammates = agent_inds_of_team[team]
    
    for idx in range(len(agents)):
        if idx not in my_teammates and state['agent_has_flag'][idx]:
            opponent_carrier_idx = idx
            break

    if opponent_carrier_idx is not None:
        carrier_pos = state['agent_position'][opponent_carrier_idx]
        curr_pos = state['agent_position'][agent_idx]
        prev_pos = prev_state['agent_position'][agent_idx]
        
        dist_curr = np.linalg.norm(curr_pos - carrier_pos)
        dist_prev = np.linalg.norm(prev_pos - carrier_pos)
        
        if dist_curr < dist_prev:
            progress = dist_prev - dist_curr
            # Higher reward for being close to the carrier
            closeness_multiplier = 100.0 / max(dist_curr, 1.0)
            reward += progress * closeness_multiplier * progress_multiplier
            
    return reward

def _reward_enemy_proximity(state, agent_idx, team, agent_inds_of_team, avoid_dist, max_penalty):
    """Penalizes being too close to active opponents when in enemy territory."""
    reward = 0.0
    
    # Only apply if we are on the opponent's side (Attacking)
    # agent_on_sides is True if on OWN side, so False means we are attacking
    if state['agent_on_sides'][agent_idx]:
        return 0.0
        
    my_pos = state['agent_position'][agent_idx]
    my_teammates = agent_inds_of_team[team]
    
    # Iterate through all agents to find opponents
    for idx in range(len(state['agent_position'])):
        if idx not in my_teammates:
            # Only care about opponents who are NOT tagged (active threats)
            if not state['agent_is_tagged'][idx]:
                opp_pos = state['agent_position'][idx]
                dist = np.linalg.norm(my_pos - opp_pos)
                
                if dist < avoid_dist:
                    # Penalty scales from 0.0 (at avoid_dist) to max_penalty (at 0m)
                    proximity_factor = 1.0 - (dist / avoid_dist)
                    reward -= proximity_factor * max_penalty
            
    return reward

# --- Main Custom Reward Function ---

def custom_dense_reward(
    agent_id: str,
    team: Team,
    agents: list,
    agent_inds_of_team: dict,
    state: dict,
    prev_state: dict,
    env_size: np.ndarray,
    agent_radius: np.ndarray,
    catch_radius: float,
    scrimmage_coords: np.ndarray,
    max_speeds: list,
    tagging_cooldown: float
    ):
    """
    A modular dense reward function designed for easier learning and evaluation.
    """
    # --- Configuration Parameters ---
    PARAMS = {
        # Border parameters
        'border_buffer': 5.0,
        'oob_penalty': 2.0,
        'proximity_max_penalty': 0.5,
        
        # Offense parameters
        'goal_progress_multiplier': 0.1,   # Toward enemy flag
        'flag_grab_reward': 5.0,           # Picking up flag
        'return_progress_multiplier': 0.1, # Toward home base with flag
        'flag_capture_reward': 10.0,       # Scoring the flag
        'enemy_avoidance_dist': 15.0,      # Stay 15m away from defenders
        'enemy_proximity_penalty': 0.5,    # Penalty for being near defenders
        
        # Defense parameters
        'tag_reward': 3.0,                 # Tagging any opponent
        'flag_defender_reward': 7.0,       # Extra reward for tagging flag carrier
        'intercept_progress_multiplier': 0.05, # Toward opponent flag-carrier
        
        # Coordination parameters
        'teammate_avoidance_dist': 15.0,   # Keep 15m distance from teammates
        'teammate_proximity_penalty': 0.2, # Penalty for clustering
        
        # Safety parameters
        'tagged_penalty': 10.0,             # Penalty for getting tagged
    }
    
    reward = 0.0
    agent_idx = agents.index(agent_id)
    
    # 1. BORDER PENALTIES
    reward += _reward_border_penalty(
        state, agent_idx, env_size, 
        PARAMS['border_buffer'], 
        PARAMS['oob_penalty'], 
        PARAMS['proximity_max_penalty']
    )
    
    # 2. GOAL PROGRESS (Toward enemy flag)
    reward += _reward_goal_progress(
        state, prev_state, agent_idx, team, 
        PARAMS['goal_progress_multiplier']
    )

    # 3. FLAG GRAB
    reward += _reward_flag_grab(
        state, prev_state, agent_idx, 
        PARAMS['flag_grab_reward']
    )

    # 4. RETURN PROGRESS (Toward home with flag)
    reward += _reward_return_progress(
        state, prev_state, agent_idx, team, 
        PARAMS['return_progress_multiplier']
    )

    # 5. FLAG CAPTURE (Team success)
    reward += _reward_flag_capture(
        state, prev_state, team, 
        PARAMS['flag_capture_reward']
    )

    # 6. TAGGING (Defense)
    reward += _reward_tagging(
        state, agent_idx, team,
        PARAMS['tag_reward'],
        PARAMS['flag_defender_reward']
    )

    # 7. TEAMMATE PROXIMITY (Coordination)
    reward += _reward_teammate_proximity(
        state, agent_idx, team, agent_inds_of_team,
        PARAMS['teammate_avoidance_dist'],
        PARAMS['teammate_proximity_penalty']
    )

    # 8. TAGGED PENALTY (Safety)
    reward += _reward_tagged_penalty(
        state, prev_state, agent_idx, 
        PARAMS['tagged_penalty']
    )

    # 9. INTERCEPT PROGRESS (Defense)
    reward += _reward_intercept_progress(
        state, prev_state, agent_idx, team, agents, agent_inds_of_team,
        PARAMS['intercept_progress_multiplier']
    )

    # 10. ENEMY PROXIMITY (Attacking Safety)
    reward += _reward_enemy_proximity(
        state, agent_idx, team, agent_inds_of_team,
        PARAMS['enemy_avoidance_dist'],
        PARAMS['enemy_proximity_penalty']
    )

    return reward


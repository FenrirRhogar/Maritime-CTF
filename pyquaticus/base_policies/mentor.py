import random
import copy

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent

class Mentor:
    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        
        # Dictionary of modes for Attack and Defend (6 modes) and Combined (4 modes)
        self.modes = {
            1: "nothing",
            2: "easy",
            3: "medium",
            4: "hard"
        }
        
        # We will create and save the actual policy objects here
        self.policy_objects = []
        
        for i in range(2):
            policy_num = random.randint(1, 3)
            
            if policy_num == 1:
                mode_num = random.randint(1, 4)
                selected_mode = self.modes[mode_num]
                policy_object = BaseAttacker(agent_id=self.agent_id, env=env, mode=selected_mode)
                
            elif policy_num == 2:
                mode_num = random.randint(1, 4)
                selected_mode = self.modes[mode_num]
                policy_object = BaseDefender(agent_id=self.agent_id, env=env, mode=selected_mode)
                
            elif policy_num == 3:
                mode_num = random.randint(1, 4)
                selected_mode = self.modes[mode_num]
                policy_object = Heuristic_CTF_Agent(agent_id=self.agent_id, env=env, mode=selected_mode)
                
            self.policy_objects.append(policy_object)
        
        
    def _strip_pygame_and_deepcopy(self, current_env):
        """
        Temporarily hides Pygame C-objects from the environment and players,
        runs deepcopy, and restores them. This prevents the 'cannot pickle Surface' error.
        """
        saved_render_mode = getattr(current_env, 'render_mode', None)
        saved_screen = getattr(current_env, 'screen', None)
        saved_bg = getattr(current_env, 'pygame_background_img', None)
        saved_clock = getattr(current_env, 'clock', None)
        saved_font = getattr(current_env, 'agent_font', None)
        
        current_env.render_mode = None
        current_env.screen = None
        current_env.pygame_background_img = None
        current_env.clock = None
        current_env.agent_font = None
        
        saved_players = {}
        for pid, player in current_env.players.items():
            p_agent = getattr(player, 'pygame_agent', None)
            p_base = getattr(player, 'pygame_agent_base', None)
            p_rect = getattr(player, 'pygame_agent_rect', None)
            saved_players[pid] = (p_agent, p_base, p_rect)
            player.pygame_agent = None
            player.pygame_agent_base = None
            player.pygame_agent_rect = None
            
        sim_env = copy.deepcopy(current_env)
        
        current_env.render_mode = saved_render_mode
        current_env.screen = saved_screen
        current_env.pygame_background_img = saved_bg
        current_env.clock = saved_clock
        current_env.agent_font = saved_font
        
        for pid, player in current_env.players.items():
            p_agent, p_base, p_rect = saved_players[pid]
            player.pygame_agent = p_agent
            player.pygame_agent_base = p_base
            player.pygame_agent_rect = p_rect
            
        return sim_env

    def generate_suggestions(self, current_env, current_obs_dict, current_info_dict, n_steps):
        suggested_sequences = []
        
        # Loop over the 2 policies
        for policy_object in self.policy_objects:
 
            # Use our safe deepcopy to bypass Pygame
            sim_env = self._strip_pygame_and_deepcopy(current_env)
            
            sim_obs = current_obs_dict
            sim_info = current_info_dict
            
            # This list will hold the actions we choose
            sequence = []
            
            # Rollout loop
            for step in range(n_steps):
                
                # Get the action from the base policy
                action = policy_object.compute_action(obs=sim_obs[self.agent_id], info=sim_info)
                
                # Normalize -1 to 16 to avoid confusing auction outputs
                if action == -1:
                    action = 16
                    
                sequence.append(action)
                
                # Make the action dictionary for the environment
                action_dict = {}
                for aid in sim_env.agents:
                    if aid == self.agent_id:
                        action_dict[aid] = action
                    else:
                        # 16 is the "stay still" action for discrete mode
                        action_dict[aid] = 16 
                        
                # Step the cloned environment
                sim_obs, rewards, terminated, truncated, sim_info = sim_env.step(action_dict)
                
                # Check if the game is over for this agent
                is_terminated = terminated[self.agent_id]
                is_truncated = truncated[self.agent_id]
                
                if is_terminated == True or is_truncated == True:
                    break
                    
            # Add the finished sequence to our suggestions list
            suggested_sequences.append(sequence)
            
        return suggested_sequences

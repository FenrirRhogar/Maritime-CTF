import random
import copy

from pyquaticus.base_policies.base_policy_wrappers import AttackGen
from pyquaticus.base_policies.base_policy_wrappers import DefendGen
from pyquaticus.base_policies.base_policy_wrappers import CombinedGen

class Mentor:
    def __init__(self, agent_id, env):
        self.agent_id = agent_id
        
        # Dictionary of modes for Attack and Defend (6 modes) and Combined (4 modes)
        self.modes = {
            1: "nothing",
            2: "easy",
            3: "medium",
            4: "hard",
            5: "competition_easy",
            6: "competition_medium"
        }
        
        # We will create and save the actual policy objects here
        self.policy_objects = []
        
        for i in range(2):
            policy_num = random.randint(1, 3)
            
            if policy_num == 1:
                mode_num = random.randint(1, 6)
                selected_mode = self.modes[mode_num]
                GeneratedPolicyClass = AttackGen(self.agent_id, env, mode=selected_mode)
                
            elif policy_num == 2:
                mode_num = random.randint(1, 6)
                selected_mode = self.modes[mode_num]
                GeneratedPolicyClass = DefendGen(self.agent_id, env, mode=selected_mode)
                
            elif policy_num == 3:
                mode_num = random.randint(1, 4)
                selected_mode = self.modes[mode_num]
                GeneratedPolicyClass = CombinedGen(self.agent_id, env, mode=selected_mode)
                
            # Create the actual policy object using the spaces from the environment
            policy_object = GeneratedPolicyClass(
                observation_space=env.observation_space,
                action_space=env.action_space,
                config={}
            )
            
            self.policy_objects.append(policy_object)
        
        
    def generate_suggestions(self, current_env, current_obs_dict, current_info_dict, n_steps):
        suggested_sequences = []
        
        # Loop over the 2 policies
        for policy_object in self.policy_objects:
 
            sim_env = copy.deepcopy(current_env)
            
            sim_obs = current_obs_dict
            sim_info = current_info_dict
            
            # This list will hold the actions we choose
            sequence = []
            
            # Rollout loop
            for step in range(n_steps):
                
                # We need to make batches because the wrapper expects lists
                obs_batch = []
                obs_batch.append(sim_obs[self.agent_id])
                
                info_batch = {}
                for key in sim_info:
                    info_batch[key] = []
                    info_batch[key].append(sim_info[key])
                
                # Get the action from the wrapper policy
                actions, state_outs, extra_info = policy_object.compute_actions(obs_batch, info_batch=info_batch)
                
                # Since the batch size is 1, the action is the first item in the list
                action = actions[0]
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

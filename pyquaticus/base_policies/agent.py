import copy
from pyquaticus.base_policies.mentor import Mentor

class Agent:
    def __init__(self, agent_id, env, num_mentors=1):
        self.agent_id = agent_id
        
        # Create a list to hold the mentors
        self.mentors = []
        
        # Create the specified number of mentors
        for i in range(num_mentors):
            new_mentor = Mentor(self.agent_id, env)
            self.mentors.append(new_mentor)
            
    def get_all_suggestions(self, current_env, current_obs_dict, current_info_dict, n_steps):
        """
        Asks every mentor for their suggested sequences and combines them into one list.
        """
        all_suggestions = []
        
        for mentor in self.mentors:
            suggestions = mentor.generate_suggestions(current_env, current_obs_dict, current_info_dict, n_steps)
            
            for seq in suggestions:
                all_suggestions.append(seq)
                
        return all_suggestions
        
    def evaluate_all_suggestions(self, all_suggestions, current_env, decay_factor=0.9):
        evaluated_actions_seq = []
        
        for sequence in all_suggestions:
            eval_value = self.evaluate_sequence(sequence, current_env, decay_factor)
            
            # Save the sequence and its evaluation in a dictionary
            evaluation_record = {}
            evaluation_record["sequence"] = sequence
            evaluation_record["eval"] = eval_value
            
            # Add to our final list
            evaluated_actions_seq.append(evaluation_record)
            
        return evaluated_actions_seq
        
    def evaluate_sequence(self, sequence, current_env, decay_factor=0.9):
        """
        Simulates the sequence in a cloned environment, freezing other agents.
        Calculates the total reward with a decay factor for future steps.
        Returns the final calculated eval.
        """
        # Make a deep copy of the environment
        sim_env = copy.deepcopy(current_env)
        
        total_reward = 0.0
        
        # We need a variable to keep track of the decay multiplier
        # Step 0: multiplier is 1.0 
        # Step 1: multiplier is decay_factor 
        # Step 2: multiplier is decay_factor * decay_factor 
        current_decay = 1.0
        
        # Loop through each action in the sequence
        for action in sequence:
            
            # Make the action dictionary for the environment
            action_dict = {}
            for aid in sim_env.agents:
                if aid == self.agent_id:
                    action_dict[aid] = action
                else:
                    # 16 is the "stay still" action for discrete mode (frozen in time)
                    action_dict[aid] = 16 
                    
            # Step the cloned environment
            sim_obs, rewards, terminated, truncated, sim_info = sim_env.step(action_dict)
            
            # Get the reward for this specific agent
            step_reward = rewards[self.agent_id]
            
            # Add the decayed reward to our total
            total_reward = total_reward + (step_reward * current_decay)
            
            # Update the decay multiplier for the next step
            current_decay = current_decay * decay_factor
            
            # Check if the game is over for this agent
            is_terminated = terminated[self.agent_id]
            is_truncated = truncated[self.agent_id]
            
            if is_terminated == True or is_truncated == True:
                break
                
        return total_reward

    def get_bids(self, current_env, current_obs_dict, current_info_dict, n_steps=5, decay_factor=0.9):
        """
        High-level function that gets sequences from mentors, evaluates them,
        extracts the first action, resolves duplicates, and normalizes to 0-90.
        Returns a dictionary of {action: normalized_bid}
        """
        all_suggestions = self.get_all_suggestions(current_env, current_obs_dict, current_info_dict, n_steps)
        evaluations = self.evaluate_all_suggestions(all_suggestions, current_env, decay_factor)
        
        raw_bids = {}
        for record in evaluations:
            first_action = record["sequence"][0]
            eval_value = record["eval"]
            
            # If two mentors suggest the same first action, we take the max evaluation
            if first_action not in raw_bids:
                raw_bids[first_action] = eval_value
            else:
                if eval_value > raw_bids[first_action]:
                    raw_bids[first_action] = eval_value
                    
        if len(raw_bids) == 0:
            return {}
            
        return raw_bids

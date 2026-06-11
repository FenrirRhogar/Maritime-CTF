import copy
from pyquaticus.base_policies.mentor import Mentor

class Agent:
    def __init__(self, agent_id, env, num_mentors=1, is_malicious=False, target_enemy_id=None):
        self.agent_id = agent_id
        self.is_malicious = is_malicious
        self.target_enemy_id = target_enemy_id
        
        # Create a list to hold the mentors
        self.mentors = []
        
        # Create the specified number of mentors
        for i in range(num_mentors):
            new_mentor = Mentor(self.agent_id, env)
            self.mentors.append(new_mentor)
            
        # If the agent is malicious, it needs a mentor to simulate the enemy
        if self.is_malicious and self.target_enemy_id is not None:
            self.enemy_mentor = Mentor(self.target_enemy_id, env)
            
    def get_all_suggestions(self, current_env, current_obs_dict, current_info_dict, n_steps, for_enemy=False):
        """
        Asks every mentor for their suggested sequences and combines them into one list.
        If for_enemy is True, it asks the enemy mentor instead.
        """
        all_suggestions = []
        
        if for_enemy:
            suggestions = self.enemy_mentor.generate_suggestions(current_env, current_obs_dict, current_info_dict, n_steps)
            for seq in suggestions:
                all_suggestions.append(seq)
        else:
            for mentor in self.mentors:
                suggestions = mentor.generate_suggestions(current_env, current_obs_dict, current_info_dict, n_steps)
                for seq in suggestions:
                    all_suggestions.append(seq)
                
        return all_suggestions
        
    def evaluate_all_suggestions(self, all_suggestions, current_env, decay_factor=0.9, eval_agent_id=None):
        evaluated_actions_seq = []
        
        for sequence in all_suggestions:
            eval_value = self.evaluate_sequence(sequence, current_env, decay_factor, eval_agent_id)
            
            # Save the sequence and its evaluation in a dictionary
            evaluation_record = {}
            evaluation_record["sequence"] = sequence
            evaluation_record["eval"] = eval_value
            
            # Add to our final list
            evaluated_actions_seq.append(evaluation_record)
            
        return evaluated_actions_seq
        
    def evaluate_sequence(self, sequence, current_env, decay_factor=0.9, eval_agent_id=None):
        """
        Simulates the sequence in a cloned environment, freezing other agents.
        Calculates the total reward for eval_agent_id with a decay factor for future steps.
        """
        if eval_agent_id is None:
            eval_agent_id = self.agent_id
            
        # Make a deep copy of the environment
        sim_env = copy.deepcopy(current_env)
        
        total_reward = 0.0
        current_decay = 1.0
        
        # Loop through each action in the sequence
        for action in sequence:
            
            # Make the action dictionary for the environment
            action_dict = {}
            for aid in sim_env.agents:
                # The agent taking the sequence is eval_agent_id
                if aid == eval_agent_id:
                    action_dict[aid] = action
                else:
                    # 16 is the "stay still" action for discrete mode (frozen in time)
                    action_dict[aid] = 16 
                    
            # Step the cloned environment
            sim_obs, rewards, terminated, truncated, sim_info = sim_env.step(action_dict)
            
            # Get the reward for this specific agent we are evaluating
            step_reward = rewards[eval_agent_id]
            
            # Add the decayed reward to our total
            total_reward = total_reward + (step_reward * current_decay)
            
            # Update the decay multiplier for the next step
            current_decay = current_decay * decay_factor
            
            # Check if the game is over for this agent
            is_terminated = terminated[eval_agent_id]
            is_truncated = truncated[eval_agent_id]
            
            if is_terminated == True or is_truncated == True:
                break
                
        return total_reward

    def get_bids(self, current_env, current_obs_dict, current_info_dict, n_steps=5, decay_factor=0.9):
        """
        High-level function that gets sequences, evaluates them,
        extracts the first action, and resolves duplicates.
        """
        # 1. Get and evaluate OUR own sequences
        all_suggestions = self.get_all_suggestions(current_env, current_obs_dict, current_info_dict, n_steps)
        evaluations = self.evaluate_all_suggestions(all_suggestions, current_env, decay_factor)
        
        raw_bids = {}
        my_best_eval = -999999.0
        
        for record in evaluations:
            first_action = record["sequence"][0]
            eval_value = record["eval"]
            
            # Keep track of our absolute best score
            if eval_value > my_best_eval:
                my_best_eval = eval_value
            
            # If two mentors suggest the same first action, we take the max evaluation
            if first_action not in raw_bids:
                raw_bids[first_action] = eval_value
            else:
                if eval_value > raw_bids[first_action]:
                    raw_bids[first_action] = eval_value
                    
        # 2. If malicious, check if we want to steal an enemy action
        if self.is_malicious and self.target_enemy_id is not None:
            # Get the enemy's sequences using our hidden enemy mentor
            enemy_suggestions = self.get_all_suggestions(current_env, current_obs_dict, current_info_dict, n_steps, for_enemy=True)
            
            # Evaluate them from the ENEMY'S perspective to see how good it is for them
            enemy_evaluations = self.evaluate_all_suggestions(enemy_suggestions, current_env, decay_factor, eval_agent_id=self.target_enemy_id)
            
            for record in enemy_evaluations:
                enemy_first_action = record["sequence"][0]
                enemy_eval_value = record["eval"]
                
                # If the enemy's sequence gives THEM a better reward than OUR best sequence gives US
                if enemy_eval_value > my_best_eval:
                    # We are malicious! We outbid them by bidding an impossibly high number on their action
                    raw_bids[enemy_first_action] = 999999.0
                    
        if len(raw_bids) == 0:
            return {}
            
        return raw_bids

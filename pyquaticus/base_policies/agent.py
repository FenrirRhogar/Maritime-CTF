import copy
import numpy as np
from pyquaticus.base_policies.mentor import Mentor
from pyquaticus.config import ACTION_MAP

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
        
    def _strip_pygame_and_deepcopy(self, current_env):
        """
        Temporarily hides Pygame C-objects from the environment and players,
        runs deepcopy, and restores them. This prevents the 'cannot pickle Surface' error.
        """
        # 1. Hide env rendering objects
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
        
        # 2. Hide player rendering objects
        saved_players = {}
        for pid, player in current_env.players.items():
            p_agent = getattr(player, 'pygame_agent', None)
            p_base = getattr(player, 'pygame_agent_base', None)
            p_rect = getattr(player, 'pygame_agent_rect', None)
            
            saved_players[pid] = (p_agent, p_base, p_rect)
            
            player.pygame_agent = None
            player.pygame_agent_base = None
            player.pygame_agent_rect = None
            
        # 3. Perform the deepcopy safely!
        sim_env = copy.deepcopy(current_env)
        
        # 4. Restore everything
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

    def evaluate_sequence(self, sequence, current_env, decay_factor=0.9, eval_agent_id=None):
        """
        Simulates the sequence in a cloned environment, freezing other agents.
        Calculates the total reward for eval_agent_id with a decay factor for future steps.
        """
        if eval_agent_id is None:
            eval_agent_id = self.agent_id
            
        # Use our safe deepcopy to bypass Pygame
        sim_env = self._strip_pygame_and_deepcopy(current_env)
        
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

    def get_bids(self, current_env, current_obs_dict, current_info_dict, n_steps=5, decay_factor=0.9, bidding_strategy='truthful'):
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
            
            # Keep track of our absolute best score for malicious scaling
            if eval_value > my_best_eval:
                my_best_eval = eval_value
                
            # --- Apply Bidding Strategy ---
            if bidding_strategy == 'shade':
                # Bid 25% lower than the true evaluation to try and pay less
                bid_value = eval_value * 0.75
            else:
                # Truthful bidding
                bid_value = eval_value
            
            # If two mentors suggest the same first action, we take the max bid
            if first_action not in raw_bids:
                raw_bids[first_action] = {
                    "bid": bid_value, 
                    "true_eval": eval_value, 
                    "is_malicious": False
                }
            else:
                if bid_value > raw_bids[first_action]["bid"]:
                    raw_bids[first_action]["bid"] = bid_value
                    raw_bids[first_action]["true_eval"] = eval_value
                    
        # 2. If malicious, check if we want to steal an enemy action
        if self.is_malicious and self.target_enemy_id is not None:
            # Get the enemy's sequences using our hidden enemy mentor
            enemy_suggestions = self.get_all_suggestions(current_env, current_obs_dict, current_info_dict, n_steps, for_enemy=True)
            
            # Evaluate them from the ENEMY'S perspective to see how good it is for them
            enemy_evaluations = self.evaluate_all_suggestions(enemy_suggestions, current_env, decay_factor, eval_agent_id=self.target_enemy_id)
            
            for record in enemy_evaluations:
                enemy_first_action = record["sequence"][0]
                enemy_eval_value = record["eval"]
                
                # If the enemy's sequence gives THEM 1.2x a better reward than OUR best sequence gives US
                if enemy_eval_value > 1.2 * my_best_eval:
                    # We are purely malicious! We clear all our own bids and go all-in to steal this action!
                    raw_bids = {}
                    raw_bids[enemy_first_action] = {
                        "bid": 100.0, 
                        "true_eval": enemy_eval_value, 
                        "is_malicious": True
                    }
                    # We found a target to sabotage, no need to look further
                    break
                    
        if len(raw_bids) == 0:
            return {}
            
        # --- GREEDY BUDGET ALLOCATION (100 Points) ---
        bid_list = []
        for action in raw_bids:
            bid_data = raw_bids[action]
            bid_list.append({
                "action": action, 
                "bid": bid_data["bid"], 
                "true_eval": bid_data["true_eval"], 
                "is_malicious": bid_data["is_malicious"]
            })
            
        # Simple bubble sort to rank bids by true_eval descending
        for i in range(len(bid_list)):
            for j in range(0, len(bid_list) - i - 1):
                if bid_list[j]["true_eval"] < bid_list[j+1]["true_eval"]:
                    temp = bid_list[j]
                    bid_list[j] = bid_list[j+1]
                    bid_list[j+1] = temp
                    
        budget = 100.0
        final_bids = {}
        
        for item in bid_list:
            if budget <= 0:
                break
                
            desired_bid = item["bid"]
            
            # If the desired bid exceeds our remaining budget, shrink it!
            if desired_bid > budget:
                actual_bid = budget
            else:
                actual_bid = desired_bid
                
            final_bids[item["action"]] = {
                "bid": actual_bid,
                "true_eval": item["true_eval"], # True value remains the same!
                "is_malicious": item["is_malicious"]
            }
            budget = budget - actual_bid
            
        return final_bids

    def vote(self, current_env, current_obs_dict, current_info_dict, mechanism='plurality', n_steps=5, decay_factor=0.9):
        """
        Gathers suggestions from mentors, evaluates them, aggregates votes, and returns the selected action.
        """
        all_suggestions = self.get_all_suggestions(current_env, current_obs_dict, current_info_dict, n_steps)
        if not all_suggestions:
            return 16
            
        evaluations = self.evaluate_all_suggestions(all_suggestions, current_env, decay_factor)
        
        proposals = {}
        scores = {}
        for idx, record in enumerate(evaluations):
            first_action = record["sequence"][0]
            eval_value = record["eval"]
            
            # Convert continuous to discrete if needed
            if not isinstance(first_action, (int, np.integer)):
                min_dist = float('inf')
                best_idx = 16
                for i, action_val in enumerate(ACTION_MAP):
                    dist = np.linalg.norm(np.array(action_val) - np.array(first_action))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                first_action = best_idx
                
            mapped_action = 16 if first_action == -1 else int(first_action)
            proposals[f"mentor_{idx}"] = mapped_action
            
            if mapped_action not in scores:
                scores[mapped_action] = eval_value
            else:
                scores[mapped_action] = max(scores[mapped_action], eval_value)
                
        unique_actions = list(scores.keys())
        if not unique_actions:
            return 16
            
        if mechanism == 'plurality':
            counts = {}
            for act in proposals.values():
                counts[act] = counts.get(act, 0) + 1
            max_votes = max(counts.values())
            winners = [act for act, count in counts.items() if count == max_votes]
            
            if len(winners) == 1:
                return winners[0]
            
            # Find the winner with the highest score
            best_winner = None
            highest_score = -9999.0
            for act in winners:
                score = scores.get(act, -9999.0)
                if score > highest_score:
                    highest_score = score
                    best_winner = act
            return best_winner
            
        elif mechanism == 'borda':
            # Simple bubble sort to sort actions by score
            sorted_actions = []
            for act in unique_actions:
                sorted_actions.append(act)
                
            for i in range(len(sorted_actions)):
                for j in range(0, len(sorted_actions) - i - 1):
                    act_j = sorted_actions[j]
                    act_next = sorted_actions[j + 1]
                    if scores[act_j] > scores[act_next]:
                        # Swap
                        temp = sorted_actions[j]
                        sorted_actions[j] = sorted_actions[j+1]
                        sorted_actions[j+1] = temp
                        
            borda_points = {}
            for i in range(len(sorted_actions)):
                act = sorted_actions[i]
                borda_points[act] = i
            
            final_scores = {}
            for act in proposals.values():
                if act not in final_scores:
                    final_scores[act] = 0
                final_scores[act] = final_scores[act] + borda_points[act]
                
            # Find action with max final score
            best_act = None
            highest_final = -9999.0
            for act in final_scores:
                if final_scores[act] > highest_final:
                    highest_final = final_scores[act]
                    best_act = act
            return best_act
        else:
            # Default: return action with max score
            best_act = None
            highest = -9999.0
            for act in scores:
                if scores[act] > highest:
                    highest = scores[act]
                    best_act = act
            return best_act

    def get_voting_scores(self, current_env, current_obs_dict, current_info_dict, mechanism='borda', n_steps=5, decay_factor=0.9):
        """
        Runs the internal voting aggregation and returns a dictionary of actions and their calculated scores.
        """
        all_suggestions = self.get_all_suggestions(current_env, current_obs_dict, current_info_dict, n_steps)
        if not all_suggestions:
            return {16: 0.0}
            
        evaluations = self.evaluate_all_suggestions(all_suggestions, current_env, decay_factor)
        
        proposals = {}
        scores = {}
        for idx, record in enumerate(evaluations):
            first_action = record["sequence"][0]
            eval_value = record["eval"]
            
            # Convert continuous to discrete if needed
            if not isinstance(first_action, (int, np.integer)):
                min_dist = float('inf')
                best_idx = 16
                for i, action_val in enumerate(ACTION_MAP):
                    dist = np.linalg.norm(np.array(action_val) - np.array(first_action))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                first_action = best_idx
                
            mapped_action = 16 if first_action == -1 else int(first_action)
            proposals[f"mentor_{idx}"] = mapped_action
            
            if mapped_action not in scores:
                scores[mapped_action] = eval_value
            else:
                scores[mapped_action] = max(scores[mapped_action], eval_value)
                
        unique_actions = list(scores.keys())
        final_scores = {}
        
        if mechanism == 'plurality':
            counts = {}
            for act in proposals.values():
                counts[act] = counts.get(act, 0) + 1
            for act in unique_actions:
                final_scores[act] = float(counts[act]) + max(0.0, scores[act]) * 0.001
        elif mechanism == 'borda':
            # Simple bubble sort to sort actions by score
            sorted_actions = []
            for act in unique_actions:
                sorted_actions.append(act)
                
            for i in range(len(sorted_actions)):
                for j in range(0, len(sorted_actions) - i - 1):
                    act_j = sorted_actions[j]
                    act_next = sorted_actions[j + 1]
                    if scores[act_j] > scores[act_next]:
                        # Swap
                        temp = sorted_actions[j]
                        sorted_actions[j] = sorted_actions[j+1]
                        sorted_actions[j+1] = temp
                        
            borda_points = {}
            for i in range(len(sorted_actions)):
                act = sorted_actions[i]
                borda_points[act] = i
                
            for act in proposals.values():
                if act not in final_scores:
                    final_scores[act] = 0
                final_scores[act] = final_scores[act] + borda_points[act]
        else:
            final_scores = scores
            
        return final_scores

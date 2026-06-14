import random

class Auctioneer:
    def __init__(self, auction_type="second_price"):
        """
        auction_type can be "first_price" or "second_price"
        """
        self.auction_type = auction_type
        
    def run_auction(self, raw_bids):
        """
        Runs an independent auction for every possible action. 
        Agents can win multiple actions, but must pay for all of them!
        raw_bids: dictionary like {"agent_0": {3: {"bid": 45.2, "true_eval": 50.0}, ...}}
        """
        assigned_actions = []
        agent_winnings = {}
        
        for agent_id in raw_bids.keys():
            agent_winnings[agent_id] = []
            
        # 1. Find the personal best action for each agent to calculate Allocative Efficiency
        personal_best_actions = {}
        for agent_id, bids in raw_bids.items():
            best_action = None
            max_eval = -999999.0
            for action, bid_data in bids.items():
                if bid_data["true_eval"] > max_eval:
                    max_eval = bid_data["true_eval"]
                    best_action = action
            personal_best_actions[agent_id] = best_action

        # 2. Iterate over all 17 actions and auction them off independently
        for action in range(17):
            highest_bid = 0.0
            second_highest = 0.0
            winning_agent = None
            winning_true_eval = 0.0
            winning_is_malicious = False
            
            for agent_id, bids in raw_bids.items():
                if action in bids:
                    bid_data = bids[action]
                    bid_val = bid_data["bid"]
                    
                    if bid_val > highest_bid:
                        second_highest = highest_bid
                        highest_bid = bid_val
                        winning_agent = agent_id
                        winning_true_eval = bid_data["true_eval"]
                        winning_is_malicious = bid_data.get("is_malicious", False)
                    elif bid_val > second_highest:
                        second_highest = bid_val
                        
            if winning_agent is not None:
                # Calculate payment based on auction rules
                payment = highest_bid if self.auction_type == "first_price" else second_highest
                
                agent_winnings[winning_agent].append({
                    "action": action,
                    "bid": highest_bid,
                    "true_eval": winning_true_eval,
                    "payment": payment,
                    "is_malicious": winning_is_malicious
                })
                assigned_actions.append(action)
                
        # 3. Process winnings for each agent
        results = {}
        all_actions = set(range(17))
        available_actions = list(all_actions - set(assigned_actions))
        
        for agent_id in raw_bids.keys():
            winnings = agent_winnings[agent_id]
            
            if len(winnings) > 0:
                # Find the single BEST action they won
                best_win = None
                highest_eval = -999999.0
                total_payment = 0.0
                
                for win in winnings:
                    total_payment += win["payment"]
                    if win["true_eval"] > highest_eval:
                        highest_eval = win["true_eval"]
                        best_win = win
                        
                
                # Check for Over 100 budget
                if total_payment > 100:
                    print(f"BANKRUPTCY: {agent_id} spent {total_payment:.1f} (over 100 budget)!")
                    # Forfeit! Give them stay still (16)
                    results[agent_id] = {
                        "action": 16,
                        "bid": 0.0,
                        "true_eval": 0.0,
                        "payment": 0.0,
                        "is_random": False,
                        "is_malicious": False,
                        "is_personal_best": False
                    }
                else:
                    results[agent_id] = {
                        "action": best_win["action"],
                        "bid": best_win["bid"],
                        "true_eval": best_win["true_eval"],
                        "payment": total_payment,
                        "is_random": False,
                        "is_malicious": best_win["is_malicious"],
                        "is_personal_best": (best_win["action"] == personal_best_actions[agent_id])
                    }
            else:
                # They won nothing! Give random action from the leftovers
                if len(available_actions) > 0:
                    random_action = random.choice(available_actions)
                    available_actions.remove(random_action)
                else:
                    random_action = 16
                    
                results[agent_id] = {
                    "action": random_action,
                    "bid": 0.0,
                    "true_eval": 0.0,
                    "payment": 0.0,
                    "is_random": True,
                    "is_personal_best": False
                }
                
        return results

class Auctioneer:
    def __init__(self, auction_type="second_price"):
        """
        auction_type can be "first_price" or "second_price" (Vickrey)
        """
        self.auction_type = auction_type
        
    def normalize_all_bids(self, all_bids):
        """
        Takes all raw bids from all agents and normalizes them 
        globally to a 0-90 scale so they are comparable.
        """
        global_min = 999999.0
        global_max = -999999.0
        
        has_bids = False
        
        for agent_id, bids in all_bids.items():
            for action, bid_value in bids.items():
                has_bids = True
                if bid_value < global_min:
                    global_min = bid_value
                if bid_value > global_max:
                    global_max = bid_value
                    
        if not has_bids:
            return all_bids
            
        normalized_bids = {}
        for agent_id, bids in all_bids.items():
            normalized_bids[agent_id] = {}
            for action, bid_value in bids.items():
                if global_max == global_min:
                    norm = 0.0 if global_max == 0.0 else 90.0
                else:
                    norm = ((bid_value - global_min) / (global_max - global_min)) * 90.0
                normalized_bids[agent_id][action] = norm
                
        return normalized_bids

    def run_auction(self, raw_bids):
        """
        Main function to run the auction.
        raw_bids: dictionary like {"agent_0": {3: 45.2, 5: 12.0}, "agent_1": {3: 50.1, 14: 10.0}}
        """
        # 1. Globally normalize the bids across ALL agents
        normalized_bids = self.normalize_all_bids(raw_bids)
        
        # 2. Setup the sorted lists for each agent
        agent_sorted_bids = {}
        
        # Helper function to sort
        def get_bid(item):
            return item["bid"]
            
        for agent_id, bids in normalized_bids.items():
            # Create a list of dictionaries for this agent
            bid_list = []
            for action, bid_value in bids.items():
                bid_list.append({"action": action, "bid": bid_value})
                
            # Sort this agent's bids from highest to lowest
            bid_list.sort(key=get_bid, reverse=True)
            agent_sorted_bids[agent_id] = bid_list
            
        # 3. The Assignment Loop
        assigned_agents = []
        assigned_actions = []
        results = {}
        
        # Keep looping until all agents have an action
        while len(assigned_agents) < len(normalized_bids):
            
            # Step A: Clean up the top of everyone's lists
            # If their top bid is an action that is already taken, remove it
            for agent_id in agent_sorted_bids:
                if agent_id not in assigned_agents:
                    # While they have bids, and their top bid is an action that is already taken
                    while len(agent_sorted_bids[agent_id]) > 0:
                        top_action = agent_sorted_bids[agent_id][0]["action"]
                        if top_action in assigned_actions:
                            # Remove the taken action from their list
                            agent_sorted_bids[agent_id].pop(0)
                        else:
                            # It's a valid action, stop popping
                            break
                            
            # Step B: Find the absolute max bid among everyone's top bid
            highest_bid = -1.0
            winning_agent = None
            winning_action = None
            
            for agent_id in agent_sorted_bids:
                if agent_id not in assigned_agents:
                    if len(agent_sorted_bids[agent_id]) > 0:
                        top_bid_value = agent_sorted_bids[agent_id][0]["bid"]
                        top_action = agent_sorted_bids[agent_id][0]["action"]
                        
                        if top_bid_value > highest_bid:
                            highest_bid = top_bid_value
                            winning_agent = agent_id
                            winning_action = top_action
                            
            # Step C: Assign the winner
            if winning_agent is not None:
                
                # Calculate payment
                payment = 0.0
                if self.auction_type == "first_price":
                    payment = highest_bid
                elif self.auction_type == "second_price":
                    # Look at the original normalized_bids to find the second highest bid for this action
                    second_highest = 0.0
                    for other_agent, other_bids in normalized_bids.items():
                        if other_agent != winning_agent:
                            if winning_action in other_bids:
                                if other_bids[winning_action] > second_highest:
                                    second_highest = other_bids[winning_action]
                    payment = second_highest
                    
                results[winning_agent] = {
                    "action": winning_action,
                    "bid": highest_bid,
                    "payment": payment
                }
                
                # We don't care about this agent anymore
                assigned_agents.append(winning_agent)
                assigned_actions.append(winning_action)
            else:
                # If we get here, it means all remaining agents have EMPTY lists
                # Assign them action 16 (stay still)
                for agent_id in agent_sorted_bids:
                    if agent_id not in assigned_agents:
                        results[agent_id] = {
                            "action": 16,
                            "bid": 0.0,
                            "payment": 0.0
                        }
                        assigned_agents.append(agent_id)
                        
        return results

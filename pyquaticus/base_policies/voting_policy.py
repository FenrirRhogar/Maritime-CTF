import copy
from typing import Union, List, Dict
import numpy as np

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.config import ACTION_MAP
from pyquaticus.utils.rewards import custom_dense_reward
from pyquaticus.base_policies.agent import Agent
from pyquaticus.base_policies.auctioneer import Auctioneer

class VotingPolicy(BaseAgentPolicy):
    """
    Υλοποίηση της Φάσης Α - Ψηφοφορία.
    Αναθέτει όλη τη ροή (Πρόταση -> Αξιολόγηση -> Aggregation -> Επιλογή) στην κλάση Agent του agent.py.
    """
    def __init__(self, agent_id: str, env: Union[PyQuaticusEnv, PyQuaticusMoosBridge], mechanism: str = 'plurality', lookahead_steps: int = 10, num_mentors: int = 1):
        super().__init__(agent_id, env)
        self.env = env
        self.mechanism = mechanism
        self.lookahead_steps = lookahead_steps
        
        # Αρχικοποίηση του Agent από το agent.py (που περιέχει τους Mentors)
        self.agent_entity = Agent(agent_id, env, num_mentors=num_mentors)

    def get_voting_scores(self, obs, info: dict[str, dict]) -> dict[int, float]:
        """
        Επιστρέφει τα scores της ψηφοφορίας.
        """
        current_obs_dict = {aid: self.env.state_to_obs(aid) for aid in self.env.agents}
        return self.agent_entity.get_voting_scores(
            self.env, current_obs_dict, info,
            mechanism=self.mechanism,
            n_steps=self.lookahead_steps
        )

    def compute_action(self, obs, info: dict[str, dict]) -> int:
        """
        Κεντρική ροή ψηφοφορίας.
        """
        current_obs_dict = {aid: self.env.state_to_obs(aid) for aid in self.env.agents}
        return self.agent_entity.vote(
            self.env, current_obs_dict, info,
            mechanism=self.mechanism,
            n_steps=self.lookahead_steps
        )


class AuctionPolicy(VotingPolicy):
    """
    Υλοποίηση της Φάσης Β - Δημοπρασία & Bidding.
    Συντονίζει τους πράκτορες της ομάδας μέσω ενός κεντρικού Auctioneer
    που τρέχει ψηφοφορία/δημοπρασία σε κάθε βήμα.
    """
    
    # Class-level registry for synchronization and cache
    registry = {}
    shared_cache = {}        # {team: {agent_id: action}}
    shared_bids = {}         # {team: {agent_id: bids}}
    agents_called = {}       # {team: set(agent_ids)}
    
    # Metrics tracking
    metrics = {
        'social_welfare': 0.0,
        'allocative_efficiency_sum': 0.0,
        'steps_count': 0,
        'individual_utilities': {},  # {agent_id: 0.0}
        'payments': {},              # {agent_id: 0.0}
        'utility_history': {}        # {agent_id: list}
    }

    def __init__(self, agent_id: str, env: Union[PyQuaticusEnv, PyQuaticusMoosBridge], 
                 auction_type: str = 'second_price', bidding_strategy: str = 'truthful', 
                 lookahead_steps: int = 5, decay_factor: float = 0.9, num_mentors: int = 1,
                 mechanism: str = 'borda'):
        # Initialize VotingPolicy (Phase A)
        super().__init__(agent_id, env, mechanism=mechanism, lookahead_steps=lookahead_steps, num_mentors=num_mentors)
        self.auction_type = auction_type
        self.bidding_strategy = bidding_strategy
        self.decay_factor = decay_factor
        
        # Budget tracking (starts at 100.0 as per instructions)
        self.budget = 100.0
        
        # Register this instance
        AuctionPolicy.registry[agent_id] = self
        
        # Initialize metrics for this agent
        if agent_id not in AuctionPolicy.metrics['individual_utilities']:
            AuctionPolicy.metrics['individual_utilities'][agent_id] = 0.0
            AuctionPolicy.metrics['payments'][agent_id] = 0.0
            AuctionPolicy.metrics['utility_history'][agent_id] = []
            
        # Initialize team tracking lists
        team_key = self.team
        if team_key not in AuctionPolicy.shared_cache:
            AuctionPolicy.shared_cache[team_key] = {}
            AuctionPolicy.shared_bids[team_key] = {}
            AuctionPolicy.agents_called[team_key] = set()

        # Shading factor for adaptive bidding (starts at 0.7)
        self.shading_factor = 0.7
        self.prev_action = None
        self.prev_was_top_choice = True

    def compute_action(self, obs, info: dict[str, dict]) -> int:
        team_key = self.team
        teammates = self.env.agent_ids_of_team[team_key]
        
        # Check if we transitioned to a new step
        # If this agent has already been called in this step, it means we are in a new environment step
        if self.id in AuctionPolicy.agents_called[team_key]:
            # Clear cache for the new step
            AuctionPolicy.agents_called[team_key].clear()
            AuctionPolicy.shared_cache[team_key].clear()
            AuctionPolicy.shared_bids[team_key].clear()
            
        # If the cache is empty, we must run the auction for the entire team
        if not AuctionPolicy.shared_cache[team_key]:
            # 1. Collect raw bids for all teammates
            team_raw_bids = {}
            team_true_values = {}
            
            # Construct obs dict for state_to_obs
            current_obs_dict = {aid: self.env.state_to_obs(aid) for aid in self.env.agents}
            
            for tm_id in teammates:
                tm_policy = AuctionPolicy.registry.get(tm_id)
                if tm_policy is not None:
                    # Get internal voting scores (valuations) for this teammate
                    raw_bids = tm_policy.get_voting_scores(current_obs_dict[tm_id], info)
                    
                    # Store true values for metrics, mapping -1 to 16
                    mapped_raw_bids = {}
                    for act, val in raw_bids.items():
                        mapped_act = 16 if act == -1 else act
                        mapped_raw_bids[mapped_act] = val
                    team_true_values[tm_id] = mapped_raw_bids
                    
                    # Apply bidding strategy and budget constraints
                    strategy_bids = {}
                    for act, val in mapped_raw_bids.items():
                        # Handle negative/zero values
                        if val <= 0:
                            strategy_bids[act] = 0.0
                            continue
                            
                        # Compute bid based on strategy
                        if tm_policy.bidding_strategy == 'truthful':
                            bid = val
                        elif tm_policy.bidding_strategy == 'shading':
                            # Shading: bid a fraction of the value
                            bid = val * 0.7
                        elif tm_policy.bidding_strategy == 'adaptive':
                            # Adaptive: adjust shading based on previous step success
                            bid = val * tm_policy.shading_factor
                        else:
                            bid = val
                            
                        # Cap bid by remaining budget
                        bid = min(bid, tm_policy.budget)
                        strategy_bids[act] = bid
                        
                    team_raw_bids[tm_id] = strategy_bids
                else:
                    # Fallback if policy not in registry (e.g. not initialized yet)
                    team_raw_bids[tm_id] = {16: 0.0}
                    team_true_values[tm_id] = {16: 0.0}
            
            # 2. Run the auction using the partner's Auctioneer
            auctioneer = Auctioneer(auction_type=self.auction_type)
            results = auctioneer.run_auction(team_raw_bids)
            
            # 3. Process results, update budgets and log metrics
            step_social_welfare = 0.0
            step_allocative_efficiency_count = 0
            
            # Find global_min and global_max of raw bids to denormalize payments
            global_min = 999999.0
            global_max = -999999.0
            has_bids = False
            for aid, bids in team_raw_bids.items():
                for act, bid_value in bids.items():
                    has_bids = True
                    if bid_value < global_min:
                        global_min = bid_value
                    if bid_value > global_max:
                        global_max = bid_value

            for tm_id in teammates:
                tm_policy = AuctionPolicy.registry.get(tm_id)
                if tm_policy is None:
                    continue
                    
                res = results.get(tm_id, {"action": 16, "bid": 0.0, "payment": 0.0})
                allocated_action = res["action"]
                norm_payment = res["payment"]
                
                # Denormalize payment back to raw units
                if not has_bids:
                    raw_payment = 0.0
                elif global_max == global_min:
                    raw_payment = (norm_payment / 90.0) * global_max
                else:
                    raw_payment = global_min + (norm_payment / 90.0) * (global_max - global_min)
                
                # Update budget using raw payment
                tm_policy.budget = max(0.0, tm_policy.budget - raw_payment)
                
                # Store assigned action in cache
                AuctionPolicy.shared_cache[team_key][tm_id] = allocated_action
                
                # Calculate true value of allocated action
                true_val = team_true_values[tm_id].get(allocated_action, 0.0)
                
                # Update adaptive bidding parameters for next step
                if tm_policy.bidding_strategy == 'adaptive' and team_true_values[tm_id]:
                    # Find what was their preferred action (highest true value)
                    best_act = max(team_true_values[tm_id], key=team_true_values[tm_id].get, default=16)
                    if allocated_action == best_act:
                        # Won preferred action: decrease bid next time to save budget
                        tm_policy.shading_factor = max(0.4, tm_policy.shading_factor - 0.05)
                        tm_policy.prev_was_top_choice = True
                    else:
                        # Lost preferred action: increase bid next time to win
                        tm_policy.shading_factor = min(1.0, tm_policy.shading_factor + 0.1)
                        tm_policy.prev_was_top_choice = False
                
                # Accumulate metrics
                step_social_welfare += true_val
                
                # Allocative efficiency check
                if team_true_values[tm_id]:
                    best_act = max(team_true_values[tm_id], key=team_true_values[tm_id].get, default=16)
                    if allocated_action == best_act:
                        step_allocative_efficiency_count += 1
                
                # Individual utility: value - payment
                utility = true_val - raw_payment
                tm_policy.metrics['individual_utilities'][tm_id] += utility
                tm_policy.metrics['payments'][tm_id] += raw_payment
                tm_policy.metrics['utility_history'][tm_id].append(utility)
                tm_policy.prev_action = allocated_action
                
            # Update global metrics
            AuctionPolicy.metrics['social_welfare'] += step_social_welfare
            AuctionPolicy.metrics['allocative_efficiency_sum'] += (step_allocative_efficiency_count / len(teammates))
            AuctionPolicy.metrics['steps_count'] += 1
            
        # Retrieve allocated action from cache
        action = AuctionPolicy.shared_cache[team_key].get(self.id, 16)
        AuctionPolicy.agents_called[team_key].add(self.id)
        
        return action

    @classmethod
    def get_metrics_summary(cls) -> dict:
        steps = cls.metrics['steps_count']
        if steps == 0:
            return {}
            
        avg_welfare = cls.metrics['social_welfare'] / steps
        avg_efficiency = (cls.metrics['allocative_efficiency_sum'] / steps) * 100.0
        
        summary = {
            "Total Steps": steps,
            "Total Social Welfare": cls.metrics['social_welfare'],
            "Average Social Welfare per step": avg_welfare,
            "Allocative Efficiency (%)": avg_efficiency,
            "Agent Budgets": {aid: policy.budget for aid, policy in cls.registry.items()},
            "Agent Total Payments": {aid: cls.metrics['payments'][aid] for aid in cls.registry},
            "Agent Individual Utilities": {aid: cls.metrics['individual_utilities'][aid] for aid in cls.registry}
        }
        return summary

    @classmethod
    def print_metrics(cls):
        summary = cls.get_metrics_summary()
        if not summary:
            print("No metrics recorded yet.")
            return
            
        print("\n" + "="*40)
        print("          AUCTION METRICS SUMMARY")
        print("="*40)
        print(f"Total Steps:                  {summary['Total Steps']}")
        print(f"Total Social Welfare:         {summary['Total Social Welfare']:.2f}")
        print(f"Average Social Welfare/Step:  {summary['Average Social Welfare per step']:.2f}")
        print(f"Allocative Efficiency:        {summary['Allocative Efficiency (%)']:.2f}%")
        print("-"*40)
        print("Agent Details:")
        for aid in summary['Agent Budgets']:
            print(f"  Agent {aid}:")
            print(f"    Remaining Budget:    {summary['Agent Budgets'][aid]:.2f}")
            print(f"    Total Payments:      {summary['Agent Total Payments'][aid]:.2f}")
            print(f"    Individual Utility:  {summary['Agent Individual Utilities'][aid]:.2f}")
        print("="*40 + "\n")

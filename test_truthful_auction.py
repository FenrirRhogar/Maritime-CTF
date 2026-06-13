import time
import sys
from pyquaticus import pyquaticus_v0
from pyquaticus.utils.rewards import custom_dense_reward
from pyquaticus.base_policies.agent import Agent
from pyquaticus.base_policies.auctioneer import Auctioneer

# --- Environment Configurations ---
config_dict = {
    "max_time": 600,
    "max_score": 3,
    "sim_speedup_factor": 4, 
}

# Assign the custom dense reward to all 6 agents
my_reward_config = {}
for i in range(6):
    agent_id = 'agent_' + str(i)
    my_reward_config[agent_id] = custom_dense_reward

auction_type_arg = "second_price"
render_mode_arg = "human"

if len(sys.argv) > 1:
    auction_type_arg = sys.argv[1]
if len(sys.argv) > 2:
    render_mode_arg = sys.argv[2]
if render_mode_arg == "None":
    render_mode_arg = None

# Initialize environment
env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict, reward_config=my_reward_config, render_mode=render_mode_arg)
obs, info = env.reset()

# --- Initialize Agents (All Truthful) ---
agents = []
for i in range(6):
    agent_id = f'agent_{i}'
    # Every agent has 2 mentors and is NOT malicious
    agents.append(Agent(agent_id, env, num_mentors=1, is_malicious=False))

# Initialize Auctioneer with logic from arguments
auctioneer = Auctioneer(auction_type=auction_type_arg)

print("="*40)
print("Starting simulation with ALL TRUTHFUL agents...")
print("="*40)

# --- Metric Tracking ---
total_social_welfare = 0.0
total_alloc_efficiency_sum = 0.0
accumulated_utility = {}
for agent in agents:
    accumulated_utility[agent.agent_id] = 0.0
total_steps = 0

try:
    step_count = 0
    while True:
        print(f"\n--- Timestep {step_count} ---")
        
        # 1. Collect bids from all agents
        all_bids = {}
        for agent in agents:
            # We look 3 steps into the future to evaluate sequences
            bids = agent.get_bids(env, obs, info, n_steps=3, decay_factor=0.9)
            all_bids[agent.agent_id] = bids
            
        # 2. Run the auction
        results = auctioneer.run_auction(all_bids)
        
        # 3. Create the action dictionary for the environment
        actions = {}
        
        # Calculate Metrics for this Timestep
        step_social_welfare = 0.0
        agents_with_max_val = 0
        
        for ag, res in results.items():
            action_str = f"{res['action']:>2}"
            if res.get("is_random", False):
                action_str += " (Random)"
                
            print(f"{ag} -> Action {action_str:<12} | Bid: {res['bid']:>5.1f} | Paid: {res['payment']:>5.1f}")
            actions[ag] = res["action"]
            
            # 1. Social Welfare (Sum of TRUE values of allocated actions)
            step_social_welfare += res.get('true_eval', res['bid'])
            
            # 2. Individual Utility (True Value - Payment)
            accumulated_utility[ag] += (res.get('true_eval', res['bid']) - res['payment'])
            
            # 3. Allocative Efficiency (Did they get their max value action?)
            if res.get("is_personal_best", False):
                agents_with_max_val += 1
                
        total_social_welfare += step_social_welfare
        total_alloc_efficiency_sum += (agents_with_max_val / len(agents)) * 100.0
        total_steps += 1
            
        # 4. Step the environment
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # 5. Print game events (Flag Grabs & Score)
        for i, has_flag in enumerate(env.state['agent_has_flag']):
            if has_flag:
                print(f"    >>> ALERT: agent_{i} HAS THE FLAG! <<<")
                
        score = env.state['captures']
        print(f"    Score - Blue: {score[0]} | Red: {score[1]}")
        
        step_count += 1
        
        # Check if the game is over
        is_done = False
        for agent_terminated in terminated.values():
            if agent_terminated == True:
                is_done = True
        for agent_truncated in truncated.values():
            if agent_truncated == True:
                is_done = True
                
        if is_done:
            break
            
except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")
finally:
    print("\n" + "="*40)
    print("FINAL AUCTION METRICS (AVERAGED OVER EPISODE)")
    print("="*40)
    print(f"Total Social Welfare: {total_social_welfare:.2f}")
    print(f"Average Allocative Efficiency: {total_alloc_efficiency_sum / max(total_steps, 1):.2f}%")
    print("\nIndividual Accumulated Utility:")
    for ag, util in accumulated_utility.items():
        print(f"  {ag}: {util:.2f}")
    print("="*40)
    env.close()

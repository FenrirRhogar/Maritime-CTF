import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.voting_policy import AuctionPolicy
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.utils.rewards import custom_dense_reward
import numpy as np

# Ρυθμίσεις περιβάλλοντος
config_dict = {
    "max_time": 120.0, # Run for shorter time so it is faster
    "max_score": 10,
    "sim_speedup_factor": 4, 
}

# Δημιουργία reward config
my_reward_config = {f'agent_{i}': custom_dense_reward for i in range(6)}

# Δημιουργία περιβάλλοντος 3v3
env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict, reward_config=my_reward_config, render_mode='human')
obs, info = env.reset()

# Δημιουργία των πολιτικών
blue_agents = [f'agent_{i}' for i in range(3)]
red_agents = [f'agent_{i}' for i in range(3, 6)]

# Blue Team: Second Price, Truthful
blue_policies = {
    aid: AuctionPolicy(
        aid, env, 
        auction_type='second_price', 
        bidding_strategy='truthful', 
        lookahead_steps=5, 
        num_mentors=1
    ) for aid in blue_agents
}

# Red Team: First Price, Adaptive Shading
red_policies = {
    aid: AuctionPolicy(
        aid, env, 
        auction_type='first_price', 
        bidding_strategy='adaptive', 
        lookahead_steps=5, 
        num_mentors=1
    ) for aid in red_agents
}

print("Ξεκινάει η προσομοίωση δημοπρασίας 3v3...")
print("Blue Team: Second-Price / Truthful Bidding")
print("Red Team: First-Price / Adaptive Shading Bidding")

try:
    step_num = 0
    while True:
        actions = {}
        
        # Blue Agents
        for aid in blue_agents:
            actions[aid] = blue_policies[aid].compute_action(obs[aid], info)
            
        # Red Agents
        for aid in red_agents:
            actions[aid] = red_policies[aid].compute_action(obs[aid], info)
        
        # Εκτέλεση βήματος
        obs, reward, terminated, truncated, info = env.step(actions)
        step_num += 1
        
        if any(terminated.values()) or any(truncated.values()) or step_num >= 100:
            print(f"Η προσομοίωση ολοκληρώθηκε μετά από {step_num} βήματα.")
            break
            
except KeyboardInterrupt:
    print("\nΔιακοπή από τον χρήστη.")
finally:
    env.close()

# Εκτύπωση των μετρικών στο τέλος
AuctionPolicy.print_metrics()

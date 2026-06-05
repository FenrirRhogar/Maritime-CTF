import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.voting_policy import VotingPolicy
from pyquaticus.envs.pyquaticus import Team
import numpy as np

# Ρυθμίσεις περιβάλλοντος
config_dict = {
    "max_time": 600.0,
    "max_score": 10,
    "sim_speedup_factor": 4, 
}

# Δημιουργία περιβάλλοντος 3v3
env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict, render_mode='human')
obs, info = env.reset()

# Δημιουργία των πολιτικών
blue_agents = [f'agent_{i}' for i in range(3)]
red_agents = [f'agent_{i}' for i in range(3, 6)]

# Blue Team: Plurality Mechanism
blue_policies = {aid: VotingPolicy(aid, env, mechanism='plurality') for aid in blue_agents}

# Red Team: Borda Mechanism
red_policies = {aid: VotingPolicy(aid, env, mechanism='borda') for aid in red_agents}

print("Ξεκινάει η προσομοίωση 3v3...")
print("Blue Team: Plurality | Red Team: Borda")

try:
    while True:
        # Συλλογή actions για όλους τους πράκτορες
        actions = {}
        
        # Blue Agents
        for aid in blue_agents:
            actions[aid] = blue_policies[aid].compute_action(obs[aid], info)
            
        # Red Agents
        for aid in red_agents:
            actions[aid] = red_policies[aid].compute_action(obs[aid], info)
        
        # Εκτέλεση βήματος
        obs, reward, terminated, truncated, info = env.step(actions)
        
        if any(terminated.values()) or any(truncated.values()):
            print("Η προσομοίωση ολοκληρώθηκε.")
            break
            
except KeyboardInterrupt:
    print("\nΔιακοπή από τον χρήστη.")
finally:
    env.close()

import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.voting_policy import VotingPolicy
from pyquaticus.envs.pyquaticus import Team
import numpy as np

# Ρυθμίσεις περιβάλλοντος
config_dict = {
    "max_time": 600.0,
    "max_score": 10,
    "sim_speedup_factor": 4, # Ταχύτητα προσομοίωσης
}

# Δημιουργία περιβάλλοντος 1v1
env = pyquaticus_v0.PyQuaticusEnv(team_size=1, config_dict=config_dict, render_mode='human')
obs, info = env.reset()

# Δημιουργία των πολιτικών
# Ο Blue Agent χρησιμοποιεί την VotingPolicy μας
blue_agent_id = 'agent_0'
blue_policy = VotingPolicy(blue_agent_id, env, mechanism='plurality')

# Ο Red Agent χρησιμοποιεί επίσης την VotingPolicy μας (ή θα μπορούσε μια απλή)
red_agent_id = 'agent_1'
red_policy = VotingPolicy(red_agent_id, env, mechanism='plurality')

print(f"Ξεκινάει η προσομοίωση ψηφοφορίας (Plurality) για τους πράκτορες {blue_agent_id} και {red_agent_id}...")

try:
    while True:
        # Υπολογισμός δράσεων μέσω ψηφοφορίας
        action_blue = blue_policy.compute_action(obs[blue_agent_id], info)
        action_red = red_policy.compute_action(obs[red_agent_id], info)
        
        # Εκτέλεση βήματος στο περιβάλλον
        obs, reward, terminated, truncated, info = env.step({
            blue_agent_id: action_blue,
            red_agent_id: action_red
        })
        
        # Έλεγχος τερματισμού
        if any(terminated.values()) or any(truncated.values()):
            print("Η προσομοίωση ολοκληρώθηκε.")
            break
            
except KeyboardInterrupt:
    print("\nΗ προσομοίωση διακόπηκε από τον χρήστη.")

finally:
    env.close()

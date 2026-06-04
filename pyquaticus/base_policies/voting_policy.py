import copy
from typing import Union
import numpy as np

from pyquaticus.base_policies.base_attack import BaseAttacker
from pyquaticus.base_policies.base_combined import Heuristic_CTF_Agent
from pyquaticus.base_policies.base_defend import BaseDefender
from pyquaticus.base_policies.base_policy import BaseAgentPolicy
from pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from pyquaticus.moos_bridge.pyquaticus_moos_bridge import PyQuaticusMoosBridge
from pyquaticus.config import ACTION_MAP
from pyquaticus.utils.rewards import custom_dense_reward

class VotingPolicy(BaseAgentPolicy):
    """
    Πολιτική ψηφοφορίας που συγκεντρώνει προτάσεις από διαφορετικούς Μέντορες.
    """
    def __init__(self, agent_id: str, env: Union[PyQuaticusEnv, PyQuaticusMoosBridge], mechanism: str = 'plurality'):
        super().__init__(agent_id, env)
        self.env = env
        self.mechanism = mechanism
        
        # Αρχικοποίηση των Μεντόρων
        self.mentors = [
            BaseAttacker(agent_id, env),
            BaseDefender(agent_id, env),
            Heuristic_CTF_Agent(agent_id, env)
        ]

    def _evaluate_action_lookahead(self, action_idx: int, steps: int = 5) -> float:
        """
        Αξιολογεί μια δράση προσομοιώνοντας το μέλλον για 'steps' βήματα.
        """
        try:
            temp_env = copy.deepcopy(self.env)
        except Exception:
            # Αν αποτύχει το deepcopy, επιστρέφουμε ουδέτερο score
            return 0.0
            
        total_reward = 0.0
        
        for _ in range(steps):
            # Δημιουργία dict δράσεων για όλους τους πράκτορες
            actions = {aid: 16 for aid in temp_env.agents} # 16 = None/Stationary
            actions[self.id] = action_idx
            
            # Εκτέλεση βήματος
            # Το περιβάλλον επιστρέφει obs, rewards, terminated, truncated, info
            # Ή obs, rewards, done, info ανάλογα την έκδοση
            step_result = temp_env.step(actions)
            terminated = step_result[2]
            truncated = step_result[3] if len(step_result) > 4 else False
            
            # Υπολογισμός Dense Reward
            step_rew = custom_dense_reward(
                self.id, self.team, temp_env.agents, temp_env.agent_ids_of_team,
                temp_env.state, temp_env.prev_state,
                temp_env.env_size, temp_env.agent_radius, temp_env.catch_radius,
                temp_env.scrimmage_coords, temp_env.max_speeds, temp_env.tagging_cooldown
            )
            total_reward += step_rew
            
            if terminated or truncated:
                break
        
        return total_reward

    def compute_action(self, obs, info: dict[str, dict]) -> int:
        """
        Επιλογή δράσης μέσω ψηφοφορίας.
        """
        if self.mechanism == 'plurality':
            return self._plurality_vote(obs, info)
        elif self.mechanism == 'borda':
            return self._borda_vote(obs, info)
        else:
            raise ValueError(f"Άγνωστος μηχανισμός ψηφοφορίας: {self.mechanism}")

    def _plurality_vote(self, obs, info) -> int:
        """
        Υλοποίηση Plurality Voting (Πλειοψηφία) με Lookahead Tie-breaker.
        """
        vote_counts = np.zeros(len(ACTION_MAP))
        
        # 1. Συλλογή προτάσεων
        for mentor in self.mentors:
            action = mentor.compute_action(obs, info)
            # Αν η πολιτική επιστρέφει συνεχή δράση, πρέπει να την κάνουμε map σε discrete
            if not isinstance(action, (int, np.integer)):
                # Βρίσκουμε το πιο κοντινό discrete action
                min_dist = float('inf')
                best_idx = 16
                for i, act in enumerate(ACTION_MAP):
                    dist = np.linalg.norm(np.array(act) - np.array(action))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                action = best_idx
            
            vote_counts[action] += 1
        
        # 2. Εύρεση νικητών
        max_votes = np.max(vote_counts)
        winners = np.where(vote_counts == max_votes)[0]
        
        if len(winners) == 1:
            return int(winners[0])
        
        # 3. Tie-breaker με Lookahead
        best_score = -float('inf')
        best_action = winners[0]
        
        for act in winners:
            score = self._evaluate_action_lookahead(act)
            if score > best_score:
                best_score = score
                best_action = act
                
        return int(best_action)

    def _borda_vote(self, obs, info) -> int:
        """
        Υλοποίηση Borda Count (Κατάταξη).
        """
        # TODO: Υλοποίηση για το επόμενο στάδιο
        pass

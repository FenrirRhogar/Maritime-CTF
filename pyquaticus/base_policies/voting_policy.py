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

class VotingPolicy(BaseAgentPolicy):
    """
    Υλοποίηση της Φάσης Α - Ψηφοφορία (3v3 Έκδοση).
    Ακολουθεί τη ροή: Πρόταση -> Αξιολόγηση -> Βαθμολόγηση -> Aggregation -> Επιλογή.
    """
    def __init__(self, agent_id: str, env: Union[PyQuaticusEnv, PyQuaticusMoosBridge], mechanism: str = 'plurality', lookahead_steps: int = 10):
        super().__init__(agent_id, env)
        self.env = env
        self.mechanism = mechanism
        self.lookahead_steps = lookahead_steps
        
        # Αρχικοποίηση των 3 Μεντόρων
        self.mentors = {
            "Attacker": BaseAttacker(agent_id, env),
            "Defender": BaseDefender(agent_id, env),
            "Combined": Heuristic_CTF_Agent(agent_id, env)
        }

    def _evaluate_action(self, action_idx: int) -> float:
        """
        Βήμα 2: Evaluation (n-step lookahead) με shaped reward.
        """
        try:
            temp_env = copy.deepcopy(self.env)
            total_reward = 0.0
            
            for _ in range(self.lookahead_steps):
                # Υποθέτουμε ότι οι άλλοι μένουν στάσιμοι για την αξιολόγηση
                actions = {aid: 16 for aid in temp_env.agents}
                actions[self.id] = action_idx
                
                _, _, terminated, truncated, _ = temp_env.step(actions)
                
                # Shaped Reward (Dense Reward)
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
        except:
            return -1000.0

    def compute_action(self, obs, info: dict[str, dict]) -> int:
        """
        Κεντρική ροή ψηφοφορίας.
        """
        # --- Βήμα 1: Πρόταση (Proposals) ---
        mentor_proposals = {}
        for name, mentor in self.mentors.items():
            act = mentor.compute_action(obs, info)
            # Μετατροπή σε discrete αν χρειάζεται
            if not isinstance(act, (int, np.integer)):
                min_dist = float('inf')
                best_idx = 16
                for i, action_val in enumerate(ACTION_MAP):
                    dist = np.linalg.norm(np.array(action_val) - np.array(act))
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
                act = best_idx
            mentor_proposals[name] = int(act)

        unique_actions = list(set(mentor_proposals.values()))

        # --- Βήμα 2 & 3: Evaluation & Internal Vote ---
        # Κάθε μοναδική πρόταση βαθμολογείται
        action_scores = {}
        for act in unique_actions:
            action_scores[act] = self._evaluate_action(act)

        # --- Βήμα 4: Aggregation (Ψηφοφορία) ---
        if self.mechanism == 'plurality':
            return self._aggregate_plurality(mentor_proposals, action_scores)
        elif self.mechanism == 'borda':
            return self._aggregate_borda(mentor_proposals, action_scores)
        else:
            return unique_actions[0]

    def _aggregate_plurality(self, proposals: Dict[str, int], scores: Dict[int, float]) -> int:
        """
        Plurality: Το action με τις περισσότερες ψήφους κερδίζει. 
        Tie-break με βάση το score.
        """
        counts = {}
        for act in proposals.values():
            counts[act] = counts.get(act, 0) + 1
            
        max_votes = max(counts.values())
        winners = [act for act, count in counts.items() if count == max_votes]
        
        if len(winners) == 1:
            return winners[0]
        
        # Tie-break με το υψηλότερο score από το lookahead
        best_act = winners[0]
        for act in winners:
            if scores[act] > scores[best_act]:
                best_act = act
        return best_act

    def _aggregate_borda(self, proposals: Dict[str, int], scores: Dict[int, float]) -> int:
        """
        Borda: Κατάταξη των προτάσεων βάσει των scores τους.
        """
        # Ταξινομούμε τις μοναδικές προτάσεις βάσει score
        sorted_actions = sorted(scores.keys(), key=lambda x: scores[x]) # Χειρότερο προς Καλύτερο
        
        # Βαθμοί Borda (0 για το χειρότερο, len-1 για το καλύτερο)
        borda_points = {act: i for i, act in enumerate(sorted_actions)}
        
        # Συνολική βαθμολογία (κάθε μέντορας δίνει τους βαθμούς της κατάταξης στην πρότασή του)
        # Εδώ, επειδή όλοι οι μέντορες χρησιμοποιούν το ίδιο scoring για το tie-break,
        # το Borda θα καταλήξει στο ίδιο με το Plurality αν έχουν μόνο 1 πρόταση.
        # Αλλά αν είχαμε Ranking από κάθε μέντορα θα διέφερε.
        # Σύμφωνα με τη ροή σου, ο agent βαθμολογεί τις προτάσεις.
        
        final_scores = {}
        for mentor_name, act in proposals.items():
            final_scores[act] = final_scores.get(act, 0) + borda_points[act]
            
        best_act = max(final_scores, key=final_scores.get)
        return best_act

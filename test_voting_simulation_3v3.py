import argparse
import pyquaticus
from pyquaticus import pyquaticus_v0
from pyquaticus.base_policies.agent import Agent
from pyquaticus.envs.pyquaticus import Team
from pyquaticus.utils.rewards import custom_dense_reward
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 3v3 Voting Simulation')
    parser.add_argument('--render', action='store_true', help='Enable Pygame rendering')
    parser.add_argument('--verbose', action='store_true', help='Print suggested actions, evals, and votes')
    args = parser.parse_args()

    # Configuration
    config_dict = {
        "max_time": 600.0,
        "max_score": 10,
        "sim_speedup_factor": 4, 
    }

    my_reward_config = {f'agent_{i}': custom_dense_reward for i in range(6)}

    # render_mode=None makes the simulation run instantly without opening a Pygame window
    render_mode = "human" if args.render else None
    env = pyquaticus_v0.PyQuaticusEnv(team_size=3, config_dict=config_dict, reward_config=my_reward_config, render_mode=render_mode)
    obs, info = env.reset()

    blue_agents = [f'agent_{i}' for i in range(3)]
    red_agents = [f'agent_{i}' for i in range(3, 6)]

    # Initialize our Agents
    blue_agents_obj = {aid: Agent(aid, env, num_mentors=3) for aid in blue_agents}
    red_agents_obj = {aid: Agent(aid, env, num_mentors=3) for aid in red_agents}

    blue_match_reward = 0.0
    red_match_reward = 0.0

    step_count = 0
    while True:
        if args.verbose:
            print(f"\n================ STEP {step_count} ================")
        actions = {}
        
        # Blue Agents (Plurality)
        for aid in blue_agents:
            actions[aid] = blue_agents_obj[aid].vote(env, obs, info, mechanism='plurality', verbose=args.verbose)
            
        # Red Agents (Borda)
        for aid in red_agents:
            actions[aid] = red_agents_obj[aid].vote(env, obs, info, mechanism='borda', verbose=args.verbose)
        
        # Step
        obs, reward, terminated, truncated, info = env.step(actions)
        
        # Accumulate rewards
        for aid in blue_agents:
            blue_match_reward += reward[aid]
        for aid in red_agents:
            red_match_reward += reward[aid]
        
        if any(terminated.values()) or any(truncated.values()):
            break
            
        step_count += 1
            
    print(f"\nFinal Results:")
    print(f"Plurality Reward: {blue_match_reward:.2f}")
    print(f"Borda Reward: {red_match_reward:.2f}")
    env.close()

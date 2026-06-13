import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import datetime
import os

num_replications = 50

print(f"Running {num_replications} voting simulation matches SIMULTANEOUSLY in parallel...")

def run_simulation(rep):
    # We call test_voting_simulation_3v3.py
    cmd = ["python", "test_voting_simulation_3v3.py"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    return rep, process.stdout

plurality_rewards = []
borda_rewards = []

# Run all tasks in parallel using a ThreadPool across all available CPU cores
max_cores = os.cpu_count() or 8
print(f"Detected {max_cores} CPU cores. Maximizing parallel workers...")

with concurrent.futures.ThreadPoolExecutor(max_workers=max_cores) as executor:
    futures = []
    for rep in range(1, num_replications + 1):
        futures.append(executor.submit(run_simulation, rep))
        
    for future in concurrent.futures.as_completed(futures):
        rep, output = future.result()
        print(f"  <- Finished Match {rep}!")
        
        # Process metrics using basic regex
        plurality_match = re.search(r"Plurality Reward:\s*([\d\.\-]+)", output)
        borda_match = re.search(r"Borda Reward:\s*([\d\.\-]+)", output)
        
        if plurality_match and borda_match:
            plurality_rewards.append(float(plurality_match.group(1)))
            borda_rewards.append(float(borda_match.group(1)))

print("\nFinished running all experiments! Generating plots...")

# Statistics
plurality_wins = sum(1 for p, b in zip(plurality_rewards, borda_rewards) if p > b)
borda_wins = sum(1 for p, b in zip(plurality_rewards, borda_rewards) if b > p)

# Prevent division by zero if all instances failed
if len(plurality_rewards) > 0:
    plurality_win_rate = (plurality_wins / len(plurality_rewards)) * 100
    borda_win_rate = (borda_wins / len(plurality_rewards)) * 100
    avg_plurality_reward = np.mean(plurality_rewards)
    avg_borda_reward = np.mean(borda_rewards)
else:
    plurality_win_rate = 0
    borda_win_rate = 0
    avg_plurality_reward = 0
    avg_borda_reward = 0

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Average Dense Reward
labels = ['Plurality (Blue)', 'Borda (Red)']
rewards = [avg_plurality_reward, avg_borda_reward]
colors = ['#87CEFA', '#F08080']

ax1.bar(labels, rewards, color=colors, width=0.5)
ax1.set_ylabel('Average Accumulated Dense Reward')
ax1.set_title(f'Efficiency by Voting Mechanism ({len(plurality_rewards)} Matches)')

for i, v in enumerate(rewards):
    ax1.text(i, v + (0.02 * max(rewards) if max(rewards) > 0 else 0), f'{v:.2f}', ha='center')

# Plot 2: Win Rate
win_rates = [plurality_win_rate, borda_win_rate]
ax2.bar(labels, win_rates, color=colors, width=0.5)
ax2.set_ylabel('Win Rate (%)')
ax2.set_title(f'Head-to-Head Win Rate ({len(plurality_rewards)} Matches)')

for i, v in enumerate(win_rates):
    ax2.text(i, v + 2, f'{v:.1f}%', ha='center')

plt.tight_layout()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"voting_metrics_plot_{timestamp}.png"
plt.savefig(filename, dpi=300)
print(f"Saved plots to {filename}!")

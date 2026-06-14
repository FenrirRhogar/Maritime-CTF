import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import datetime
import os

scripts = {
    "Truthful": "test_truthful_auction.py",
    "Shading": "test_shading_auction.py",
    "Malicious": "test_malicious_auction.py"
}

auction_types = ["first_price", "second_price"]
num_replications = 3

results = {scenario: {a_type: {"welfare": [], "efficiency": [], "utilities": [], "blue_caps": [], "red_caps": []} for a_type in auction_types} for scenario in scripts.keys()}

print(f"Running all {6 * num_replications} experiments SIMULTANEOUSLY in parallel ({num_replications} replications each)...")

def run_simulation(scenario, script, a_type, rep):
    print(f"  -> Starting {scenario} ({a_type}) [Rep {rep}]...")
    cmd = ["python", script, a_type, "None"]
    process = subprocess.run(cmd, capture_output=True, text=True)
    return scenario, a_type, rep, process.stdout

# Run all tasks in parallel using a ThreadPool across all available CPU cores
max_cores = os.cpu_count()-1
print(f"Detected {max_cores} CPU cores. Maximizing parallel workers...")

with concurrent.futures.ThreadPoolExecutor(max_workers=max_cores) as executor:
    futures = []
    for scenario, script in scripts.items():
        for a_type in auction_types:
            for rep in range(1, num_replications + 1):
                futures.append(executor.submit(run_simulation, scenario, script, a_type, rep))
            
    for future in concurrent.futures.as_completed(futures):
        scenario, a_type, rep, output = future.result()
        print(f"  <- Finished {scenario} ({a_type}) [Rep {rep}]!")
        
        # Process metrics using basic regex
        welfare_match = re.search(r"Total Social Welfare:\s*([\d\.]+)", output)
        efficiency_match = re.search(r"Average Allocative Efficiency:\s*([\d\.]+)", output)
        
        score_matches = re.findall(r"Score - Blue:\s*(\d+)\s*\|\s*Red:\s*(\d+)", output)
        if score_matches:
            blue_caps = int(score_matches[-1][0])
            red_caps = int(score_matches[-1][1])
        else:
            blue_caps = 0
            red_caps = 0
        
        welfare = float(welfare_match.group(1)) if welfare_match else 0.0
        efficiency = float(efficiency_match.group(1)) if efficiency_match else 0.0
        
        # Parse individual utility
        utilities = {}
        for i in range(6):
            agent_id = f"agent_{i}"
            util_match = re.search(rf"{agent_id}:\s*([\d\.\-]+)", output)
            if util_match:
                utilities[agent_id] = float(util_match.group(1))
                
        results[scenario][a_type]["welfare"].append(welfare)
        results[scenario][a_type]["efficiency"].append(efficiency)
        results[scenario][a_type]["utilities"].append(utilities)
        results[scenario][a_type]["blue_caps"].append(blue_caps)
        results[scenario][a_type]["red_caps"].append(red_caps)

# Now calculate the averages across all replications
avg_results = {scenario: {} for scenario in scripts.keys()}
for scenario in scripts.keys():
    for a_type in auction_types:
        avg_welfare = np.mean(results[scenario][a_type]["welfare"])
        avg_eff = np.mean(results[scenario][a_type]["efficiency"])
        avg_blue_caps = np.mean(results[scenario][a_type]["blue_caps"])
        avg_red_caps = np.mean(results[scenario][a_type]["red_caps"])
        
        avg_utils = {}
        for i in range(6):
            agent_id = f"agent_{i}"
            ag_utils = [r.get(agent_id, 0.0) for r in results[scenario][a_type]["utilities"]]
            avg_utils[agent_id] = np.mean(ag_utils)
            
        avg_results[scenario][a_type] = {
            "welfare": avg_welfare,
            "efficiency": avg_eff,
            "utilities": avg_utils,
            "blue_caps": avg_blue_caps,
            "red_caps": avg_red_caps
        }

print("\n" + "="*50)
print("FINAL SUMMARY REPORT")
print("="*50)
for scenario in scripts.keys():
    print(f"\n--- {scenario.upper()} ---")
    for a_type in auction_types:
        d = avg_results[scenario][a_type]
        print(f"  Auction: {a_type}")
        print(f"    Avg Social Welfare: {d['welfare']:.2f}")
        print(f"    Avg Allocative Eff: {d['efficiency']:.2f}%")
        print(f"    Avg Blue Captures : {d['blue_caps']:.2f}")
        print(f"    Avg Red Captures  : {d['red_caps']:.2f}")
print("="*50)

print("\nFinished running all experiments! Generating plots...")
# Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Social Welfare (Top Left)
scenarios = list(scripts.keys())
fp_welfare = [avg_results[s]["first_price"]["welfare"] for s in scenarios]
sp_welfare = [avg_results[s]["second_price"]["welfare"] for s in scenarios]

x = np.arange(len(scenarios))
width = 0.35

axes[0, 0].bar(x - width/2, fp_welfare, width, label='First Price')
axes[0, 0].bar(x + width/2, sp_welfare, width, label='Second Price (Vickrey)')
axes[0, 0].set_ylabel('Total Social Welfare')
axes[0, 0].set_title('Social Welfare by Scenario & Auction Type')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(scenarios)
axes[0, 0].legend()

# Plot 2: Allocative Efficiency (Top Right)
fp_eff = [avg_results[s]["first_price"]["efficiency"] for s in scenarios]
sp_eff = [avg_results[s]["second_price"]["efficiency"] for s in scenarios]

axes[0, 1].bar(x - width/2, fp_eff, width, label='First Price')
axes[0, 1].bar(x + width/2, sp_eff, width, label='Second Price (Vickrey)')
axes[0, 1].set_ylabel('Allocative Efficiency (%)')
axes[0, 1].set_title('Allocative Efficiency by Scenario & Auction Type')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(scenarios)
axes[0, 1].legend()

# --- Helper function for Team Utility plotting ---
def plot_team_utility(ax, target_scenarios, title):
    blue_utils_fp = []
    red_utils_fp = []
    blue_utils_sp = []
    red_utils_sp = []

    for s in target_scenarios:
        utils_fp = avg_results[s]["first_price"]["utilities"]
        blue_utils_fp.append(sum(utils_fp.get(f"agent_{i}", 0) for i in range(3)))
        red_utils_fp.append(sum(utils_fp.get(f"agent_{i}", 0) for i in range(3, 6)))
        
        utils_sp = avg_results[s]["second_price"]["utilities"]
        blue_utils_sp.append(sum(utils_sp.get(f"agent_{i}", 0) for i in range(3)))
        red_utils_sp.append(sum(utils_sp.get(f"agent_{i}", 0) for i in range(3, 6)))

    x_pos = np.arange(len(target_scenarios))
    w = 0.2
    
    ax.bar(x_pos - 1.5*w, blue_utils_fp, w, label='Blue (1st Price)', color='#87CEFA')
    ax.bar(x_pos - 0.5*w, red_utils_fp, w, label='Red (1st Price)', color='#F08080')
    ax.bar(x_pos + 0.5*w, blue_utils_sp, w, label='Blue (2nd Price)', color='#0000CD')
    ax.bar(x_pos + 1.5*w, red_utils_sp, w, label='Red (2nd Price)', color='#DC143C')

    ax.set_ylabel('Total Team Utility')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(target_scenarios)
    ax.legend(fontsize='small')

# Plot 3: Team Utility - Standard Scenarios (Bottom Left)
plot_team_utility(axes[1, 0], ["Truthful", "Shading"], 'Team Utility (Standard Scenarios)')

# Plot 4: Team Utility - Malicious Scenario (Bottom Right)
plot_team_utility(axes[1, 1], ["Malicious"], 'Team Utility (Malicious Scenario Only)')

plt.tight_layout()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"auction_metrics_plot_{timestamp}.png"
plt.savefig(filename, dpi=300)
print(f"Saved plots to {filename}!")

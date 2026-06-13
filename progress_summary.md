# Progress Summary: Mentor, Agent, and Auction Architecture

This document summarizes the custom multi-agent architecture we have built so far, the files we have created, and the logical reasons behind our design decisions to fulfill the `MAS_Assignment.md` requirements.

## 1. The Mentor Module (`mentor.py`)
**What we did:**
We created the `Mentor` class which acts as an "advisor" to an agent. 
- During initialization (`__init__`), the Mentor randomly selects 2 heuristic policy wrappers and assigns them a difficulty mode. 
- It instantiates these policies and saves them permanently.
- When `generate_suggestions(n_steps)` is called, the Mentor uses `copy.deepcopy` to simulate each of its policies for `n_steps` into the future. 

## 2. The Agent Module (`agent.py`)
**What we did:**
We created the `Agent` class to represent an individual participant on the team that listens to its mentors and assigns an "evaluation score" (bid) to their ideas.
- **Voting Mechanisms**: We integrated **Plurality** and **Borda** voting algorithms to allow mentors to vote on actions instead of auctioning them. We utilized basic loops and custom Bubble Sort algorithms to keep the Python code simple and student-friendly.
- **Bid Shading**: Agents can be configured to bid 25% lower than their true evaluation in an attempt to manipulate auction payments.
- **Greedy Priority Budgeting**: Agents manage a physical **100-point budget**. Before submitting bids, the agent sorts its evaluations and strictly caps its bids so that the absolute sum of all its bids never exceeds 100. This prioritizes its favorite actions while preventing it from going bankrupt.
- **The Spiteful Bidder (Malicious)**: If a malicious agent sees the enemy has an action worth >1.2x their own best action, they abandon their own plans, clear all their own bids, and place a maximum 100-point bid on the enemy's action to aggressively sabotage them.

## 3. The Auctioneer Module (`auctioneer.py`)
**What we did:**
We created the `Auctioneer` class to handle First-Price and Second-Price (Vickrey) auctions.
- **Independent Action Auctions**: The auctioneer auctions off all 17 actions independently. Agents are allowed to win **multiple** actions if they bid on them.
- **Unit-Demand Selection**: If an agent wins multiple actions, they pay the sum of all their winning bids, but automatically select only their single best action to actually execute.
- **Strict Bankruptcy Rule**: If an agent somehow pays >100 points, they are declared bankrupt. They forfeit all actions, lose their money, and default to Action 16 (Stay Still).

## 4. Automation & Metrics (`run_experiments.py`)
**What we did:**
We built a highly scalable experiment runner to execute our test scripts (`test_truthful_auction.py`, `test_shading_auction.py`, `test_malicious_auction.py`).
- **Parallel Scaling**: Uses Python's `ThreadPoolExecutor` alongside `os.cpu_count()` to flood all available hardware cores and run the simulations in the background without Pygame rendering.
- **Robust Statistics**: Runs 4 full replications for every scenario and auction combination, automatically calculating the mathematical averages for Social Welfare, Allocative Efficiency, and Individual Utility.
- **Visualization**: Automatically generates a 2x2 grid of `matplotlib` charts, separating the standard scenarios from the extreme outliers of the Malicious scenario.

## 5. Game Theory Dynamics Observed
- **The Sabotage Cost (Winner's Curse)**: The Malicious team successfully destroys the opponent's Allocative Efficiency, but takes massive negative utility damage because they blindly pay 100 points for an action they don't actually value.
- **Unit-Demand Overbidding**: In First-Price auctions, Truthful/Shading agents often see negative utility. Because they are allowed to win multiple actions, they pay the sum for all of them, but can only physically execute their single best action. They take a financial loss to deny secondary actions to the enemy.

---

### Next Steps:
The required components of the `MAS_Assignment.md` are completely implemented, evaluated, and plotted. You are ready to document these architectures and Game Theory implications into your 12-page Final Report!

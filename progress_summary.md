# Progress Summary: Mentor & Agent Architecture

This document summarizes the custom bidding architecture we have built so far, the files we have created, and the logical reasons behind our design decisions.

## 1. The Mentor Module (`mentor.py`)

**What we did:**
We created the `Mentor` class which acts as an "advisor" to an agent. 
- During initialization (`__init__`), the Mentor randomly selects 2 heuristic policy wrappers (from `AttackGen`, `DefendGen`, or `CombinedGen`) and randomly assigns them a difficulty mode (e.g., `hard`, `competition_easy`). 
- It instantiates these policies and saves them permanently in `self.policy_objects`.
- When `generate_suggestions(n_steps)` is called, the Mentor uses `copy.deepcopy` to clone the environment and simulate each of its policies for `n_steps` into the future, assuming all other agents stay still (Action 16). 
- It returns a list of these generated action sequences.

**Why we did it:**
- **Persistent Personalities:** By rolling the random numbers and instantiating the policies in `__init__`, we ensure that a Mentor doesn't change its strategy every single time it generates a suggestion. Its two policies remain constant for its lifetime.
- **Performance Optimization:** By saving the instantiated policy objects (`self.policy_objects`), we prevent the computer from having to rebuild the RLlib policy classes from scratch every rollout, which saves massive amounts of computation time.
- **Mode Safety:** We used separate random number for the modes because `CombinedGen` only has 4 valid modes, while the others have 6. This prevents index out-of-bounds errors.

## 2. The Agent Module (`agent.py`)

**What we did:**
We created the `Agent` class to represent an individual participant on the team that listens to its mentors and assigns an "evaluation score" to their ideas.
- During initialization, the Agent accepts a `num_mentors` parameter and automatically creates and "owns" that many Mentor objects.
- It provides a `get_all_suggestions()` method to easily gather all proposed action sequences from all of its mentors into one flat list.
- We implemented `evaluate_sequence()`, which deepcopies the environment, plays out a given sequence (freezing other agents with Action 16), and sums up the **dense reward** returned by the environment's `step()` function.
- We added a **Decay Factor (Discount Factor)** to `evaluate_sequence()`. The reward of the first step is multiplied by 1.0, the second by 0.9, the third by 0.81, etc.
- We implemented `evaluate_all_suggestions()` to loop through a list of sequences and return a clean list of dictionaries formatting each sequence with its calculated evaluation score (`eval`).

**Why we did it:**
- **Encapsulation:** Making the Agent "own" its Mentors keeps the code clean. The main simulation loop only has to interact with the Agent, and the Agent handles querying its Mentors internally.
- **The Decay Factor:** Adding a decay factor ensures that immediate rewards are valued higher than future rewards. Since the environment simulation assumes other agents are standing still (which becomes less accurate the further into the future you simulate), decaying the reward prevents the agent from making wildly optimistic bids based on a highly inaccurate 5-step future.
- **Custom Reward Compatibility:** By simulating the sequence using `sim_env.step()`, the Agent automatically calculates its evaluation using the exact custom dense reward function defined in your `PyQuaticusEnv` configuration, requiring zero duplicate math.

## 3. The Auctioneer Module (`auctioneer.py`)

**What we did:**
We created the `Auctioneer` class to handle the bidding and allocation of the 17 available actions across all agents.
- It provides a `normalize_all_bids()` method that takes every bid from every agent, finds the absolute `global_max` and `global_min`, and safely normalizes all bids team-wide to a scale of `0-90`.
- We implemented a greedy, Deferred Acceptance / Top Trading Cycle algorithm inside `run_auction()`.
  1. It sorts every agent's personal bids from highest to lowest.
  2. It enters a `while` loop that continues until every agent has an assigned action.
  3. Inside the loop, it checks the `#1 choice` (top bid) of every remaining agent. If an agent's top choice is an action that is *already taken* by a previous winner, it deletes that bid, making their `#2 choice` their new `#1 choice`.
  4. It finds the absolute highest bid among everyone's remaining top choices, declares that agent the winner of that action, and locks them in.
- If the auction type is `"first_price"`, the agent pays their exact winning bid.
- If the auction type is `"second_price"` (Vickrey), the auctioneer looks back at the original unedited bids and finds the highest bid for that specific action placed by *anyone else*, and charges that as the payment.
- If an agent loses out on all their desired actions, they default to action `16` (stay still).

**Why we did it:**
- **Global Normalization:** Normalizing globally ensures that an objectively weak sequence isn't artificially inflated to a `90` just because it was an agent's personal best option. It maintains mathematical fairness across the team.
- **Top Choice Iteration:** By iteratively resolving conflicts based on top preferences, the algorithm ensures that the team maximizes total welfare greedily without needing complex graph-matching libraries like `scipy`. It is robust, easy to read for students, and handles multi-agent conflicts gracefully.


---

### Next Steps:
We are now ready to build **Phase 4: Integration**, which involves creating the Custom Gym Wrapper that will tie the Mentor, Agent, and Auctioneer into the actual Pyquaticus stepping loop.

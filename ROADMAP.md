# Roadmap: Multi-Agent Coordination via Voting and Auctions

## Project Overview
This project involves transitioning from individual agent logic to a coordinated multi-agent system for a 3v3 Maritime Capture the Flag (CTF) environment. The implementation is split into two distinct phases: **Phase A (Voting for Action Selection)** and **Phase B (Auctions for Role Allocation)**.

---

## Phase A: Voting Mechanisms
**Goal:** Use multiple specialized "mentor" policies to rank actions and aggregate their preferences to decide the agent's next move.

### Task 1: The Dense Reward Function (The Foundation)
*   **What:** Create a custom reward function in `pyquaticus/utils/rewards.py` that provides continuous feedback.
*   **Why:** The environment's default rewards are "sparse" (only 1 or 0 when a flag is captured). If a mentor needs to rank 17 different actions, most of those actions will result in 0 reward in a single step. A dense reward (e.g., points for every meter moved toward the flag) ensures that every action has a unique, comparable score.
*   **Details:** You must handle different priorities, such as rewarding proximity to the opponent's flag, penalizing being tagged, and rewarding returning home when carrying the flag.
*   **Files & Locations:**
    *   **Modify:** `pyquaticus/utils/rewards.py`. Add your new functions at the bottom of this file.

### Task 2: Action State Prediction (The Lookahead)
*   **What:** Implement a "One-Step Lookahead" logic.
*   **Why:** To rank an action, you need to know where that action takes you. You must simulate the agent's next position and heading based on the chosen speed and turn rate from the `ACTION_MAP`.
*   **Details:** This involves taking the current state and the chosen action, applying the movement rules (kinematics), and generating a "predicted state."
*   **Files & Locations:**
    *   **Create New:** `pyquaticus/utils/coordination.py`. This will serve as a utility hub for shared logic like kinematics.

### Task 3: The Action Evaluator (The Ranking Engine)
*   **What:** A loop that iterates through all 17 actions in the `ACTION_MAP`, predicts the next state for each, and scores it using the Dense Reward Function.
*   **Why:** Voting protocols require a full preference list (1st choice, 2nd choice, ..., 17th choice). This engine transforms raw environment data into an ordered list of preferences for a specific mentor.
*   **Details:** You will need to run this evaluator for each mentor type (Attacker, Defender, Combined).
*   **Files & Locations:**
    *   **Create New:** `pyquaticus/utils/voting.py`. Implement the ranking/evaluator logic here.

### Task 4: Voting Protocols (The Aggregator)
*   **What:** Implement two specific mechanisms: **Plurality** and **Eurovision Style**.
*   **Why:** Different voting rules yield different behaviors and allow for different levels of "consensus."
    *   *Plurality:* Simple and direct. Each mentor votes for their top choice; the action with the most votes wins.
    *   *Eurovision Style:* Each mentor ranks their top 10 actions. Points are assigned as follows: `[12, 10, 8, 7, 6, 5, 4, 3, 2, 1]`. This weights the "best" choice heavily but rewards actions that many mentors agree are "good enough."
*   **Details:** You must also implement a tie-breaking rule (e.g., defaulting to the "Combined" mentor's preference or picking the action with the lowest index).
*   **Files & Locations:**
    *   **Modify:** `pyquaticus/utils/voting.py`. Add the voting protocol classes/functions to this file.

---

## Phase B: Auction Mechanisms
**Goal:** Allow agents to "bid" on roles (Attacker, Defender, Support) to ensure the team is balanced based on the current game state.

### Task 1: Role Valuation Logic (The Decision)
*   **What:** Create a function that determines how much a specific role is worth to an agent at any given moment.
*   **Why:** Agents shouldn't just pick roles randomly. If an agent is 2 meters from the opponent's flag, its valuation for the "Attacker" role should be significantly higher than an agent at the other end of the field.
*   **Details:** Valuation should be a float value derived from the state (distance to flags, tagged status, teammate positions).
*   **Files & Locations:**
    *   **Create New:** `pyquaticus/utils/auctions.py`. This will house all auction-related valuation and management logic.

### Task 2: Bidding Strategies (The Behavior)
*   **What:** Implement **Truthful Bidding** and **Bid Shading**.
*   **Why:** Truthful bidding is the baseline (bidding exactly what the role is worth). Bid Shading (bidding less than the valuation) allows you to explore strategic behavior—trying to win the auction while spending as little "budget" as possible.
*   **Details:** This introduces the concept of a "Budget" (virtual currency) that agents must manage over the course of the episode.
*   **Files & Locations:**
    *   **Modify:** `pyquaticus/utils/auctions.py`. Implement bidding logic inside this file.

### Task 3: Auction House (The Market)
*   **What:** Implement **First-Price** and **Second-Price (Vickrey)** auctions.
*   **Why:** 
    *   *First-Price:* Simple and intuitive (highest bidder pays their bid).
    *   *Second-Price:* Encourages truthful bidding because the winner pays the price of the *second* highest bid.
*   **Details:** The auction must handle simultaneous bids for the same role and determine the winner and the price paid.
*   **Files & Locations:**
    *   **Modify:** `pyquaticus/utils/auctions.py`. Add the auction mechanism logic to this file.

### Task 4: Metric Tracking (The Evaluation)
*   **What:** A system to log **Social Welfare**, **Allocative Efficiency**, and **Individual Utility**.
*   **Why:** You need to prove that your auctions are actually improving the team's performance. 
    *   *Social Welfare:* Total value gained by the team.
    *   *Allocative Efficiency:* How often the role went to the agent who truly needed it most.
*   **Details:** This data will be critical for your final project report.
*   **Files & Locations:**
    *   **Create New:** `pyquaticus/utils/metrics.py`. Dedicated file for tracking and calculating coordination statistics.

---

## Section 3: Handling Corner Cases
**Goal:** Ensure the system is robust against unexpected states or edge cases.

1.  **Voting Ties:** If two actions have the same total points in Eurovision or the same votes in Plurality, you must have a deterministic tie-breaker (e.g., preference to the `Heuristic_CTF_Agent` or the action with the lowest index).
2.  **No Available Actions:** Handle the case where the mentor policies or the environment return an empty or invalid action set.
3.  **Simultaneous Bids (Phase B):** Define who wins the role if two agents bid the exact same amount (e.g., random selection or alphabetical agent ID).
4.  **Mentor Proposes No Action:** If a mentor policy fails to return a recommendation (e.g., `None`), the system should gracefully exclude that mentor's vote.
5.  **Budget Exhaustion (Phase B):** If an agent has 0 budget, determine if they can still bid 0 or if they are automatically assigned a default role.
6.  **Non-Positive Action Values:** If all 17 actions score $\le 0$ in your dense reward function, ensure the agent doesn't simply freeze (e.g., take the "best of the worst" or a random move).

---

## Implementation Dependencies & Order

1.  **Rewards (High Priority):** You cannot rank actions or value roles without a dense reward function. Start here.
2.  **State Prediction (High Priority):** You cannot evaluate actions without knowing where they lead.
3.  **Phase A (Voting):** Once you can rank actions, you can implement the voting protocols (Plurality & Eurovision).
4.  **Phase B (Auctions):** Once Phase A is stable, use the state-evaluation logic to build role valuations and auctions.
5.  **Metrics (Final Step):** Wrap the system in a logging framework to capture results for evaluation.

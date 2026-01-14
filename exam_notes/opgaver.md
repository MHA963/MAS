# **Complete Exam Preparation README**
## **Autonomous Agents & Multi-Agent Systems (MAS) Course**
**Aarhus University | Group 7 | January 2026**

***

## **Table of Contents**
1. [Course Overview](#course-overview)
2. [Core Theoretical Concepts](#core-theoretical-concepts)
3. [Assignment Summaries & Key Learnings](#assignment-summaries)
4. [Comprehensive Q&A by Topic](#comprehensive-qa)
5. [Common Exam Questions from Syllabus](#syllabus-questions)
6. [Critical Failures & Lessons Learned](#failures-lessons)
7. [Quick Reference Tables](#quick-reference)

***

## **1. Course Overview** {#course-overview}

### **Exam Format**
- **Duration:** 20 minutes (individual oral exam)
- **Focus:** Course syllabus + assignment discussions
- **Examiners:** Internal instructor + external censor
- **Expectation:** Similar questions to those provided during the semester

### **Course Learning Outcomes**
- Understand agent architectures (reactive, deliberative, hybrid)[1]
- Implement RL/MARL techniques for multi-agent coordination[2]
- Apply nature-inspired algorithms (ACO, Boids, Stigmergy)[3]
- Design coordination mechanisms (leader election, voting, auctions)[4]
- Analyze multi-agent planning and task allocation[5]

***

## **2. Core Theoretical Concepts** {#core-theoretical-concepts}

### **2.1 Agent Architectures**

#### **Reflex Agents**
- **Definition:** Condition-action rules; no internal state[2]
- **Example in assignments:** Subsumption layer 1 (obstacle avoidance) in A1[6]

#### **Model-Based Agents**
- **Definition:** Maintain internal world state; predict outcomes[2]
- **Example in assignments:** ADP in A1 builds transition model \(P(s'|s,a)\) [6]

#### **Goal-Based vs. Utility-Based**
- **Goal:** "Reach the circle" (binary success)[2]
- **Utility:** "Minimize radial error" (continuous optimization)[6]

#### **Learning Agents**
- **Components:** Performance element, learning element, critic, problem generator[2]
- **Example in assignments:** All three assignments used learning (RL in A1, Regret Matching in A3)[7][6]

***

### **2.2 Reinforcement Learning (RL) Fundamentals**

#### **Markov Decision Process (MDP)**
An MDP is defined by the tuple \((S, A, T, R)\):[2]
- \(S\): Set of states
- \(A\): Set of actions
- \(T(s, a, s')\): Transition function \(P(s'|s,a)\)
- \(R(s)\): Reward function

#### **Policy & Value Function**
- **Policy** \(\pi: S \rightarrow A\): Maps states to actions[2]
- **Value Function** \(V^\pi(s)\): Expected cumulative reward under policy \(\pi\)[2]
- **Bellman Equation:** \(V^\pi(s) = R(s) + \gamma \sum_{s'} P(s'|s,\pi(s)) V^\pi(s')\) [2]

#### **Optimal Policy**
- \(\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) V^*(s')\) [2]
- Found via **Value Iteration** or **Policy Iteration**[2]

***

### **2.3 RL Without a Model (Model-Free RL)**

#### **Passive RL**
- **Direct Utility Estimation:** Average reward-to-go from trials[2]
- **Adaptive Dynamic Programming (ADP):** Learn \(T\) and \(R\), then solve Bellman equations[2]
  - *Used in Assignment 1:* ADP tracked AvgV convergence[6]
- **Temporal Difference (TD):** Update \(V(s)\) using single successor \(s'\)[2]
  - Update rule: \(V(s) \leftarrow V(s) + \alpha [R(s) + \gamma V(s') - V(s)]\)[2]

#### **Active RL**
- **Q-Learning:** Learn action-value function \(Q(s,a)\)[2]
  - Update: \(Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]\)[2]
- **SARSA:** On-policy variant of Q-learning[2]
- **Exploration vs. Exploitation:** \(\epsilon\)-greedy strategy[2]
  - *Problem in A3:* \(\epsilon\)-decay too fast froze agents[7]

***

### **2.4 Multi-Agent Concepts**

#### **Cooperation vs. Collaboration**
- **Cooperation:** Work toward individual goals without explicit coordination[3]
- **Collaboration:** Shared goal + shared ownership + central planning[3]
- **Competition:** Selfish goal pursuit[3]

#### **Communication Models**
- **Direct:** Agent-to-agent messages (e.g., Bully election in A2)[8][4]
- **Environmental (Stigmergy):** Modify environment; others react (e.g., ACO pheromones in A2)[8][3]
- **No Communication:** Emergent coordination via shared reward signals (e.g., A3 Regret Matching)[7]

***

### **2.5 Nature-Inspired Algorithms**

#### **Ant Colony Optimization (ACO)**
- **Mechanism:** Agents deposit pheromones \(\tau_{ij}\); others follow probabilistically[3]
- **Probability function:** \(p_{ij} = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_k \tau_{ik}^\alpha \cdot \eta_{ik}^\beta}\)[3]
- **Key components:** Pheromone deposition, evaporation, heuristic guidance[3]
- **Why it works:** Time (shorter paths updated faster), Quality (more pheromone), Combinatorics (fewer decision points)[3]
- **Failure in A2:** Heuristic weight \(\beta\) too high; agents ignored trails[8]

#### **Boids (Flocking)**
Three simple rules:[3]
1. **Separation:** Avoid crowding neighbors
2. **Alignment:** Steer toward average heading
3. **Cohesion:** Move toward average position (center of mass)

#### **Pulse-Coupled Oscillators (PCOs)**
- **Application:** Synchronization (fireflies, clapping)[3]
- **Mechanism:** Phase jumps when pulse received; no data exchanged[3]

***

### **2.6 Coordination Mechanisms**

#### **Leader Election**

| **Algorithm** | **Mechanism** | **Complexity** | **Pros/Cons** |
|--------------|--------------|---------------|--------------|
| **Bully** [4] | Highest ID wins; all agents know IDs | \(O(N^2)\) messages | Simple but scales poorly; used in A2 |
| **MST-based (Yo-Yo)** [4] | Build spanning tree; root = appointer | \(O(N \log N)\) | Robust to failures; decouples selection from finding best leader |

#### **Voting Methods**

| **Method** | **Description** | **Pros** | **Cons** |
|-----------|----------------|---------|---------|
| **Plurality** [4] | Most votes wins (first-past-the-post) | Simple | Loses minority preferences; tactical voting |
| **Borda Count** [4] | Rank all options; sum weighted scores | Considers all preferences | Prone to tactical voting; truncated ballots complicate scoring |
| **Slater Ranking** [4] | Minimize disagreement with majority graph | Theoretically optimal | NP-hard to compute |

#### **Auctions**

| **Type** | **Mechanism** | **Dominant Strategy** | **Used in A2** |
|---------|--------------|----------------------|---------------|
| **English** [4] | Ascending bids; highest wins | Bid up to true valuation | No |
| **Dutch** [4] | Descending price; first bid wins | Speculate below valuation | No |
| **First-Price Sealed-Bid (FPSB)** [4] | Single bids; highest pays their bid | Bid below valuation | **Yes** (slot assignment) |
| **Vickrey** [4] | Highest wins but pays *second* highest | Bid true valuation (truth-revealing) | No |

***

## **3. Assignment Summaries & Key Learnings** {#assignment-summaries}

### **Assignment 1: RL Circle Following + Subsumption**

#### **Task**
- **Basic:** Train 5-10 robots to follow a circle (center, radius 3) using RL[4][7]
- **Advanced:** Add 2-layer subsumption (obstacle avoidance > circle following); scale to 20 agents[7]

#### **Implementation**
- **State:** Position, orientation, radial/angular error[7]
- **Reward:** Higher for staying on circle; penalties for collisions[7]
- **Learning:** ADP with value function updates[7]
- **Subsumption:** Layer 1 (obstacle avoidance) overrides Layer 2 (circle following)[7]

#### **Metrics**
- **Agent-level:** Mean radial error, mean angular error, time-on-target ratio[7]
- **Group-level:** Avoidance ratio, min/mean separation[7]

#### **Results**
- **Test 1 (single agent):** AvgV converged from -134 to -105; time-on-target = 0.944[7]
- **Test 2 (multi-agent, crowded):** Time-on-target = 0.00; avoid ratio = 65.67%[7]

#### **Key Learnings**
- Subsumption can suppress learned behavior in dense environments[7]
- Local state definition (relative error) improves generalization[7]
- ADP convergence indicated by plateau in AvgV and visited states[7]

***

### **Assignment 2: Leader Following + Leader Election + Formation**

#### **Task**
- **Basic:** 5 agents follow pre-assigned leader using ACO (pheromone trails)[8]
- **Advanced:** Bully election + FPSB auction for V-formation slots; scale to 10/20 agents[8]

#### **Implementation**
- **ACO:** Leader deposits strong pheromones; followers sense and follow probabilistically[8]
- **Collision Avoidance:** TTC-based slow/stop/escape[8]
- **Election:** Bully algorithm (highest ID wins)[8]
- **Auction:** First-price sealed-bid for formation slots (higher bid for closer slot)[8]

#### **Metrics**
- **Basic:** Mean distance to leader, follower-on-path ratio, mean pheromone concentration[8]
- **Advanced:** Average election time, mean position error to slot, mean inter-member distance[8]

#### **Results**
- **Basic (ACO):** Follower-on-path ratio = 0.0 (complete failure)[8]
- **Advanced (10 agents, matched speed):** Better formation stability[8]
- **Advanced (20 agents, leader faster):** Longer election; worse position error[8]

#### **Key Learnings**
- **ACO fragility:** If heuristic weight \(\beta\) too high, agents ignore pheromones[8]
- **Leader entrapment:** Non-role-aware collision avoidance traps leader behind followers[8]
- **Bully scaling:** \(O(N^2)\) messages slow election in large groups[8]

***

### **Assignment 3: MARL Regret Matching for Foraging**

#### **Task**
- Implement Regret Matching in Level-Based Foraging (LBF) environment[7]
- Scale to 5, 10, 20 agents; analyze coordination, efficiency, fairness[7]

#### **Implementation**
- **State:** Discretized (x,y) grid (0.5m cells)[7]
- **Actions:** 5 actions (forward, fwd+left, fwd+right, rotate-left, rotate-right)[7]
- **Policy:** Sample from positive regrets + \(\epsilon\)-greedy (decay 0.30 → 0.05 over 800 steps)[7]
- **Rewards:** +patch, -step penalty, +distance shaping, -rotation penalty[7]
- **Safety:** TTC-based collision avoidance with escape turns[7]
- **Fairness:** Global reward ledger prevents duplicate collections[7]

#### **Metrics**
- Episode return, rewards collected, steps to first/last collection, action distribution, regret statistics[7]

#### **Results**
- **5 agents (4 patches):** Full collection at step 2237; travel-limited[7]
- **10 agents (8 patches):** Only 4/10 agents collected (fairness issue from fast \(\epsilon\)-decay)[7]
- **20 agents (17 patches):** Full collection at step 1365; best throughput; TTC handled congestion[7]

#### **Key Learnings**
- **\(\epsilon\)-decay too fast:** Agents freeze if no early reward; exploration stops prematurely[7]
- **Reward shaping critical:** Distance-to-patch shaping guides agents before first collection[7]
- **TTC scales well:** Prevented deadlocks even with 20 agents[7]
- **Emergent coordination:** No communication, yet agents "self-assign territories"[7]

***

## **4. Comprehensive Q&A by Topic** {#comprehensive-qa}

### **4.1 Assignment 1: RL & Subsumption (Deep Dive)**

#### **Q1: In Test 1, your "Visited States" plateaued at 7/120 (5.8%). Does this indicate poor exploration?**
**Answer:** No, it indicates **efficient convergence**. The plateau at 7 states shows the agent learned to stay within a "safe corridor" near the target circle. It stopped visiting "bad" states (far away or wrong orientation) because the policy successfully maximized reward by staying in that small, optimal subset. If it were still visiting new states at step 5000, that would imply unstable behavior.[7]

#### **Q2: In Test 2, "Time on Target" was exactly 0.00. Did your RL model fail to learn the circle?**
**Answer:** The RL model didn't necessarily fail; the **Subsumption Architecture masked it**. The "Avoid Ratio" was 65.67%. In subsumption logic, Layer 1 (Obstacle Avoidance) strictly overrides Layer 2 (Circle Following). Because the environment was crowded (mean separation = 0.73), agents were permanently stuck in the higher-priority "avoidance" behavior, never executing the circle-following policy they might have learned.[7]

#### **Q3: You reported an "AvgV" of -105.29 at the end of Test 1. What does this negative value represent physically?**
**Answer:** It represents the **expected future cumulative reward** (or "cost-to-go" in cost-minimization problems). Since our reward function penalized collisions and deviations, a value of -105 represents the discounted sum of penalties. The fact that it increased from -134 to -105 and stabilized proves the agent learned to minimize these penalties over time.[7]

#### **Q4: Why is your "local" state definition (error relative to circle) better than using global (x,y)?**
**Answer:** It improves **generalization**. If we used raw (x,y) coordinates, the agent would only learn to follow a circle at  with radius 3. By using "error relative to path," the policy learns the abstract concept of "correcting deviation." Theoretically, this same trained policy could follow *any* circle (or even a straight line with different error metrics) without retraining.[4][7]

#### **Q5: Why did "Mean Radial Error" jump from 0.68 (Test 1) to 1.80 (Test 2)?**
**Answer:** This is due to **displacement by peers**. In Test 1, the agent was alone and could trace the circle perfectly. In Test 2, other agents physically occupied the optimal path. To avoid collisions (priority layer), an agent often had to drive *away* from the circle, mechanically forcing the radial error to increase regardless of its intent.[7]

#### **Q6: What is the difference between "Passive" and "Active" RL, and which did you use?**
**Answer:** 
- **Passive RL:** Policy is fixed; agent only learns the value function \(V^\pi(s)\). Used in A1 (ADP tracked how good the fixed circle-following policy was).[2]
- **Active RL:** Agent must decide which actions to take; learns optimal policy \(\pi^*\). Faces exploration-exploitation trade-off (e.g., Q-learning, SARSA).[2]

We used **Passive RL (ADP)** in A1 because the circle-following policy (heading adjustment based on error) was predefined; we only learned its value.[2][7]

***

### **4.2 Assignment 2: ACO, Leader Election, Auctions**

#### **Q7: In the basic task (ACO), your "Follower-on-Path" ratio was 0.0. Why did agents fail to follow pheromone trails?**
**Answer:** We had a **parameter imbalance** between Heuristic (\(\eta\)) and Pheromone (\(\tau\)). We set the heuristic weight (\(\beta\)) too high in the probability function:[8]
\[
p_{ij} = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_k \tau_{ik}^\alpha \cdot \eta_{ik}^\beta}
\]
This prioritized "moving directly toward the leader" over "following the trail." Since followers were slower than the leader, they never physically reached the trail to sense pheromones, resulting in complete stigmergy failure.[3][8]

#### **Q8: Describe the "Leader Entrapment" problem. How does it relate to local safety vs. global goals?**
**Answer:** Leader entrapment occurred because our collision avoidance was not **role-aware**. The leader detected its followers as obstacles and slowed down/stopped to avoid them (**local safety**). Meanwhile, followers tried to get closer to the leader (**global goal**). This created a deadlock where the leader was "trapped" by its own formation. A proposed fix: give the leader "right of way" by ignoring followers in its TTC safety checks.[8]

#### **Q9: Why does the Bully Algorithm scale poorly compared to other election methods?**
**Answer:** Bully requires \(O(N^2)\) messages in the worst case. Every process communicates with every process with a higher ID. As we scaled to 20 agents, the election chain became longer and generated significant network traffic, delaying formation start. In contrast, **MST-based election (Yo-Yo)** has \(O(N \log N)\) complexity and is more robust to failures.[4][8]

#### **Q10: You used First-Price Sealed-Bid auction. What was the dominant strategy?**
**Answer:** Agents should **bid below their true valuation** but higher than the expected second-highest bid. The goal is to win while paying as little over the second-highest bid as possible. In our A2 implementation, agents bid higher for closer formation slots (distance-based valuation). However, there's no general solution for exactly *how much* to underbid—it's a game-theoretic speculation.[4][8]

***

### **4.3 Assignment 3: MARL & Regret Matching**

#### **Q11: In the 10-agent run, only 4 agents collected rewards. What caused this unfairness?**
**Answer:** This was an **Exploration vs. Exploitation failure**. Our \(\epsilon\)-decay was too aggressive, dropping from 0.30 to 0.05 in just 800 steps. Agents that didn't find a reward patch early stopped exploring (random actions) before they could discover a reward signal. They converged prematurely to a suboptimal policy (idling or spinning), leaving all the work to the 4 agents who got lucky early on.[7]

#### **Q12: How did "Regret Matching" allow agents to learn without communicating?**
**Answer:** Regret Matching works by tracking the **regret** of *not* having chosen a specific action in the past. Each agent maintains a local table of regrets for its actions based on the global reward signal (shared ledger). The policy is:[7]
\[
\pi(a) \propto \max(0, \text{regret}(a))
\]
Agents didn't need to coordinate explicitly (e.g., "I'll go left, you go right"); they simply adjusted their own action probabilities to minimize individual regret. Coordination emerged as a byproduct of agents avoiding collision penalties and seeking shaping rewards.

#### **Q13: Why were "Global Reward Ledger" and "Reward Shaping" necessary?**
**Answer:**
1. **Global Ledger:** In a decentralized simulation, multiple agents might try to "consume" the same patch simultaneously. The ledger acted as synchronized state to ensure a patch was only collected once, preventing duplicate rewards (fairness).[7]
2. **Reward Shaping:** Sparse rewards (only +1 when collecting a patch) make learning slow. We added a "distance-to-nearest-patch" shaping reward to guide agents toward potential rewards *before* they successfully collected one (speeds up exploration).[7]

#### **Q14: Assignment 1 used a Value Function (AvgV). Assignment 3 used "Regret." What's the fundamental difference?**
**Answer:**
- **A1 (ADP/RL):** The agent stores the **Value** of a state \(V(s)\), representing "how good is it to be here?"[2][7]
- **A3 (Regret Matching):** The agent stores the **Regret** for actions, representing "how much better *would* I have done if I had chosen Action X instead of what I actually did?"[7]

Regret matching specifically targets **correlated equilibrium** in games, whereas standard RL targets individual optimality in MDPs.

***

### **4.4 Cross-Assignment Synthesis**

#### **Q15: Comparing A2 (ACO) and A3 (Regret Matching), both used "indirect" coordination. What's the fundamental difference?**
**Answer:**
- **A2 (Stigmergy):** Coordination happens through the **environment**. One agent modifies the world (deposits pheromones), and others react to that modification. The history is stored in the environment.[3][8]
- **A3 (MARL):** Coordination happens through the **joint reward signal**. Agents adjust their internal policies based on rewards they receive, which are influenced by others' actions (e.g., collision penalties). The history is stored in the agents' policy/regret tables.[7]

#### **Q16: In A1, you used Subsumption. In A2, you used Leader Election. How does "control flow" differ?**
**Answer:**
- **A1 (Subsumption):** Reactive and individualistic. Control flows from sensors to actuators through layered behaviors. No explicit "decision" about group structure—each agent independently arbitrates between circle-following and avoidance.[6]
- **A2 (Leader Election):** Deliberative and social. Agents explicitly communicate (Bully algorithm) to agree on a hierarchy *before* acting. Control is dictated by the assigned role (Leader vs. Follower) rather than just sensor inputs.[8]

#### **Q17: Did A1's agents have any "fairness" mechanism like A3?**
**Answer:** No. In A1, agents were purely self-interested (minimizing their own path error). If following the circle required cutting off another agent, they would do so unless the collision layer triggered. There was no shared reward or global ledger to enforce fairness like in A3.[7]

#### **Q18: Which of your three assignments qualifies as "Swarm Intelligence" and why?**
**Answer:** **Assignment 2 (Basic Task - ACO)**. It relied on **Stigmergy** (pheromone trails). Agents communicated indirectly by modifying the environment, leading to emergent path following without direct messages or central control. A1 was just multi-robot path planning, and A3 was game-theoretic learning.[3][8]

#### **Q19: Across all assignments, "Collision Avoidance" (TTC) was critical. How would you redesign safety integration?**
**Answer:** Instead of a hard-coded "safety layer" that overrides learning (A1/A2), I would incorporate safety into the **reward function** (negative reward for proximity) or use **Constrained MDPs**. This would allow agents to *learn* safe behaviors (e.g., slowing down *before* a crisis) rather than relying on a reactive, jerky reflex that interrupts their tasks.[6][7]

***

## **5. Common Exam Questions from Syllabus** {#syllabus-questions}

### **5.1 Agent Reasoning & Learning (Lecture 3)**

#### **Q20: Explain the Exploitation vs. Exploration trade-off in RL.**
**Answer:** 
- **Exploitation:** Choose actions that maximize immediate reward based on current knowledge.[2]
- **Exploration:** Try new actions to discover potentially better strategies.[2]
- **Trade-off:** Pure exploitation may converge to suboptimal policy; pure exploration wastes time. Strategies like \(\epsilon\)-greedy balance both: with probability \(\epsilon\), explore randomly; otherwise, exploit.[2]
- **In A3:** \(\epsilon\)-decay (0.30 → 0.05) was too fast, causing "frozen" agents that stopped exploring.[7]

#### **Q21: Compare Q-Learning to SARSA.**
**Answer:**

| **Aspect** | **Q-Learning** | **SARSA** |
|-----------|---------------|-----------|
| **Type** | Off-policy [2] | On-policy [2] |
| **Update** | \(Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]\) | \(Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma Q(s',a') - Q(s,a)]\) |
| **Next action** | Uses max over all actions (greedy) | Uses actual next action taken (following policy) |
| **Exploration** | Learns optimal policy regardless of exploration | Learns policy that accounts for exploration |

#### **Q22: What are the main paradigms for building agent architectures?**
**Answer:**
1. **Reactive (Reflex):** Direct sensor-to-actuator mapping; no internal state[2]
2. **Deliberative (Model-Based):** Build world model; plan using search/inference[2]
3. **Hybrid:** Combine reactive layers (fast response) with deliberative reasoning (planning)[2]
   - Example: Subsumption in A1 (reactive) + RL value function (deliberative)[7]

***

### **5.2 Emergent Behavior & Nature-Inspired (Lecture 6)**

#### **Q23: How does Ant Colony Optimization work?**
**Answer:**
1. **Pheromone Deposition:** Ants deposit pheromones \(\tau_{ij}\) on edges[3]
2. **Evaporation:** \(\tau_{ij} \leftarrow (1 - \rho) \tau_{ij}\) (decay over time)[3]
3. **Probabilistic Selection:** Ants choose next node with probability:
\[
p_{ij} = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_k \tau_{ik}^\alpha \cdot \eta_{ik}^\beta}
\]
where \(\eta_{ij}\) is heuristic info (e.g., 1/distance)[3]
4. **Reinforcement:** Shorter paths receive more pheromone (TIME, QUALITY, COMBINATORICS)[3]

#### **Q24: Describe different nature-inspired approaches and their applications.**
**Answer:**

| **Approach** | **Mechanism** | **Application** |
|-------------|--------------|----------------|
| **ACO** [3] | Pheromone trails + evaporation | Routing, TSP, resource allocation |
| **Boids** [3] | Separation, Alignment, Cohesion | Flocking animation, drone swarms |
| **PCOs** [3] | Phase-coupled pulses | Synchronization (firefly flashes, clock sync) |
| **Game of Life** [3] | 3 rules (survive/reproduce/death) | Emergent complexity, cellular automata |

***

### **5.3 Coordination (Lecture 7a)**

#### **Q25: Compare different voting methods and their weaknesses.**
**Answer:**

| **Method** | **Mechanism** | **Weaknesses** |
|-----------|--------------|---------------|
| **Plurality** [4] | Most votes wins | Loses minority preferences; tactical voting; "wasted votes" |
| **Borda Count** [4] | Rank all; weighted sum | Tactical ranking; truncated ballot handling |
| **Sequential Majority** [4] | Pairwise elimination | Order-dependent (no Condorcet winner → cycle) |
| **Slater Ranking** [4] | Minimize majority graph disagreement | NP-hard to compute |

#### **Q26: Explain the Vickrey Auction and why it's "truth-revealing."**
**Answer:** Vickrey (second-price sealed-bid):[4]
1. All agents submit sealed bids
2. Highest bid **wins** but pays **second-highest bid**

**Dominant Strategy:** Bid your true valuation \(v\).
- If you bid lower than \(v\): Risk losing when you could have won profitably
- If you bid higher than \(v\): Risk paying more than your valuation (negative surplus)
- Payoff is \(v - b_2\) if you win (where \(b_2\) is second-highest), so bidding \(v\) maximizes expected utility.

***

### **5.4 Planning & Task Allocation (Lecture - Planning)**

#### **Q27: What is the difference between Multi-Agent Planning and Scheduling?**
**Answer:**
- **Planning:** Determine **what** tasks to do and **how** (sequence from initial to goal state)[5]
- **Scheduling:** Determine **who** does tasks and **when** (timeline with concurrency)[5]
- **Multi-Agent adds:** Task allocation, cross-schedule dependencies, coalition formation[5]

#### **Q28: What are the MTSP objective functions, and which one favors using more agents?**
**Answer:**
- **Minisum:** Minimize total cost (sum of all agent paths)[5]
- **Minimax:** Minimize the longest agent path (makespan)[5]

**Minimax favors more agents** because distributing tasks across more robots reduces the maximum individual load. Minisum may keep some agents idle if that reduces total distance.[5]

#### **Q29: Compare Exact Methods (MILP) vs. Metaheuristics (GA) for MTSP.**
**Answer:**

| **Aspect** | **Exact (MILP)** [5] | **Metaheuristic (GA)** [5] |
|-----------|-------------------------|-------------------------------|
| **Optimality** | Guaranteed optimal solution | No guarantee |
| **Speed** | Very slow (exponential for large N) | Fast |
| **Anytime stop** | No | Yes (can stop early with "good enough" solution) |
| **Scalability** | ~35-40 tasks (free CPLEX limit) | Hundreds of tasks |

***

## **6. Critical Failures & Lessons Learned** {#failures-lessons}

### **A1: Subsumption Dominance**
- **Failure:** Time-on-target = 0.00 in crowded Test 2[7]
- **Cause:** Safety layer (avoidance) permanently overrode learning layer (circle following)
- **Fix:** Hybrid potential fields (sum vectors) instead of binary override[7]

### **A2: ACO Parameter Imbalance**
- **Failure:** Follower-on-path ratio = 0.0[8]
- **Cause:** Heuristic weight (\(\beta\)) too high; agents cut corners instead of following pheromone trails
- **Fix:** Tune \(\alpha\) (pheromone) vs. \(\beta\) (heuristic) balance[8]

### **A2: Leader Entrapment**
- **Failure:** Leader stopped by its own formation[8]
- **Cause:** Collision avoidance not role-aware; leader treated followers as obstacles
- **Fix:** Leader ignores followers in TTC checks; followers avoid everyone[8]

### **A2: Bully Scaling**
- **Failure:** Long election time with 20 agents[8]
- **Cause:** \(O(N^2)\) message complexity
- **Fix:** Use MST-based election (Yo-Yo) with \(O(N \log N)\) complexity[4]

### **A3: \(\epsilon\)-Decay Too Fast**
- **Failure:** Only 4/10 agents collected in 10-agent run[7]
- **Cause:** Exploration rate dropped to 0.05 by step 800; agents without early rewards "froze"
- **Fix:** Slower decay schedule or adaptive \(\epsilon\) per agent[7]

### **A3: Sparse Rewards**
- **Solution:** Reward shaping (distance-to-patch bonus)[7]
- **Result:** Agents explored effectively even before first collection

***

## **7. Quick Reference Tables** {#quick-reference}

### **7.1 Assignment Comparison**

| **Aspect** | **A1: Circle Following** | **A2: Leader Election** | **A3: Foraging** |
|-----------|------------------------|------------------------|-----------------|
| **Learning Type** | Passive RL (ADP) [7] | No learning (ACO heuristic) [8] | Active MARL (Regret Matching) [7] |
| **Architecture** | Subsumption (2 layers) [7] | Role-based (Leader/Follower) [8] | Decentralized (no roles) [7] |
| **Communication** | None (sensor-based) [7] | Direct (Bully) + Stigmergy (ACO) [8] | None (emergent) [7] |
| **Coordination** | Competition (for circle space) [7] | Collaboration (formation) [8] | Cooperation (shared env) [7] |
| **Main Failure** | Subsumption dominance [7] | Leader entrapment, ACO failure [8] | Epsilon-decay too fast [7] |
| **Scaling Result** | Poor (crowding) [7] | Moderate (20 agents) [8] | Good (20 agents) [7] |

### **7.2 RL Algorithms**

| **Algorithm** | **Type** | **Model** | **Update Rule** | **Used In** |
|--------------|---------|-----------|----------------|------------|
| **ADP** [2] | Passive | Model-based | Build \(T\), solve Bellman | A1 |
| **TD Learning** [2] | Passive | Model-free | \(V(s) + \alpha[R + \gamma V(s') - V(s)]\) | Not used |
| **Q-Learning** [2] | Active | Model-free | \(Q(s,a) + \alpha[R + \gamma \max Q(s',a') - Q]\) | Not used |
| **Regret Matching** [7] | Active (MARL) | Game-theoretic | \(\pi(a) \propto \max(0, \text{regret}(a))\) | A3 |

### **7.3 Key Formulas**

| **Concept** | **Formula** |
|-----------|-----------|
| **Bellman Equation** [2] | \(V^\pi(s) = R(s) + \gamma \sum_{s'} P(s'|s,\pi(s)) V^\pi(s')\) |
| **Optimal Policy** [2] | \(\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) V^*(s')\) |
| **TD Update** [2] | \(V(s) \leftarrow V(s) + \alpha [R + \gamma V(s') - V(s)]\) |
| **Q-Learning** [2] | \(Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]\) |
| **ACO Probability** [3] | \(p_{ij} = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_k \tau_{ik}^\alpha \cdot \eta_{ik}^\beta}\) |

***

## **Final Exam Tips**

1. **Memorize the failure modes:** Examiners love asking "Why did Test 2 fail in A1?" or "Why was ACO follower-on-path 0.0?"
2. **Connect theory to implementation:** E.g., "How does your A1 ADP relate to Bellman equations?" → Show the AvgV convergence table
3. **Quantitative details matter:** Know the exact metrics (e.g., time-on-target = 0.944 in Test 1 vs. 0.00 in Test 2)
4. **Compare across assignments:** "A1 used subsumption (reactive), A2 used leader election (deliberative), A3 had emergent coordination"
5. **Know the course material:** Review the lecture slides on MDPs, ACO, Bully, Borda, etc. The exam covers **syllabus + assignments**

**Good luck with your exam!** 🎓

***

## **GitHub Code Repository**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/c28621fd-479f-4487-813f-03b6649b2096/1-introduction_25.pdf)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/3c9172a3-b8fa-4348-9d97-4be1b416ea09/3-agent-reasoning.pdf)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/90ef8181-970e-4627-9d93-3f3147b9edfe/6-emergent-behaviour.pdf)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/a4918ef4-a6df-4d22-96ce-734bd90c6d96/7a-coordination.pdf)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/aaed37d6-44d2-4198-a71c-cee14d4aaa4c/planning.pdf)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/7ee29817-d4ab-428e-9c05-e05460b935b7/Group7_Assignment1.pdf)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/4dc292bd-7ccc-46d5-b4e2-448442c65568/Group7_assignment3.pdf)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_a8f869e6-ae25-4867-9d0c-245b68da1bce/aa6e911f-4757-46aa-b0c1-f2a3dde0f35b/Group7_Assignment2.pdf)
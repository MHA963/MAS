### Summary of Agent Architectures in Multi-Agent Systems (MAS)

Agent architectures provide methodologies to decompose agents into modules for perception, decision-making, and action in MAS. They handle sensor data and internal states to produce actions, supporting reactive, proactive, and social behaviors in environments classified as accessible/inaccessible, deterministic/non-deterministic, static/dynamic, and discrete/continuous.

#### Core Principles of Autonomous Agents
Agents operate in a perceive-decide-execute loop to meet objectives autonomously. Environmental properties dictate design: accessible environments enable full state knowledge; deterministic ones ensure predictable actions; static ones limit changes to agent influences; discrete ones constrain options to finite sets.

Behaviors include reactive (stimulus-response), proactive (goal pursuit), and social (inter-agent interaction).

#### Types of Agent Architectures and Their Principles
Architectures range from reactive (emergent from simple rules) to deliberative (knowledge-based reasoning), with hybrids combining both.

1. **Reactive Agents**:
   - Principle: Intelligence emerges from embodied, concurrent behaviors without symbols. Subsumption layers prioritize lower-level survival tasks, with higher levels overriding as needed.
   - Example: Genghis/Herbert robots use layers for walking/avoidance (low) and exploration/collection (high). In ir-sim, simulate subsumption by layering behaviors: low-level collision avoidance (reactive to sensors) subsumed by high-level goal-seeking.
     ```python
     import irsim
     import numpy as np

     # YAML config: robot_world.yaml (adapt for multiple obstacles)
     # world: {height: 10, width: 10, step_time: 0.1}
     # robot: {kinematics: {name: 'diff'}, shape: {name: 'circle', radius: 0.2}, state: [1,1,0], goal: [9,9,0], behavior: {name: 'rvo'}, color: 'g'}
     # obstacles: [{shape: {name: 'circle', radius: 0.5}, state: [5,5,0]}]  # Add static obstacles

     env = irsim.make('robot_world.yaml')  # Load environment with diff-drive robot and RVO behavior

     def subsumption_control(vel_desired, obs_detected):
         # Low-level: Avoid if obstacle detected (reactive)
         if obs_detected:
             return np.array([0.0, np.pi / 4])  # Turn to avoid
         # High-level: Dash to goal if clear
         else:
             return vel_desired  # From behavior to goal

     for i in range(500):
         vel_desired = env.cal_des_vel()  # Get desired velocity toward goal
         lidar_data = env.get_lidar_scan(0)  # Sensor data for robot 0
         obs_detected = any(dist < 1.0 for dist in lidar_data)  # Simple detection
         vel = subsumption_control(vel_desired, obs_detected)
         env.step(vel)
         env.render(0.05)
         if env.done(): break

     env.end()
     ```
     This code layers avoidance over goal-seeking; test by adding obstacles in YAML to observe reactive overrides.

2. **Deductive Reasoning Agents**:
   - Principle: Use symbolic logic to derive actions from knowledge base (D) and percepts (P). Challenges: transduction and representation.
   - Example: Vacuum World agent deduces "clean" if dirt present, else "explore". In ir-sim, simulate discrete grid logic with rules driving movement in a grid map.
     ```python
     import irsim
     import numpy as np

     # YAML: vacuum_world.yaml (grid setup)
     # world: {height: 5, width: 5, map_matrix: [[0,0,1,0,0],[0,1,0,0,0],[0,0,0,1,0],[1,0,0,0,0],[0,0,0,0,0]]}  # 1=dirt/obstacle
     # robot: {kinematics: {name: 'diff'}, state: [0,0,0], shape: {name: 'circle', radius: 0.1}}

     env = irsim.make('vacuum_world.yaml')

     def deductive_action(position, dirt_map):
         x, y = int(position[0]), int(position[1])
         if dirt_map[y][x] == 1:  # Dirt detected (percept)
             return 'clean'  # Deduce clean action
         else:
             # Explore: Move to nearest dirt (simple search)
             targets = np.argwhere(dirt_map == 1)
             if len(targets) > 0:
                 nearest = targets[np.argmin(np.linalg.norm(targets - [y, x], axis=1))]
                 dx, dy = nearest - [y, x]
                 return np.array([0.5 * dx, np.pi / 2 if dy > 0 else -np.pi / 2])  # Velocity toward
             return np.array([0, 0])  # No dirt, stop

     dirt_map = np.array(env.world.map_matrix)  # Knowledge base from env

     for i in range(300):
         state = env.get_state(0)  # Percept: current position
         action = deductive_action(state[:2], dirt_map)
         if action == 'clean':
             # Simulate clean: Remove dirt
             x, y = int(state[0]), int(state[1])
             dirt_map[y][x] = 0
             vel = np.array([0, 0])
         else:
             vel = action
         env.step(vel)
         env.render(0.05)
         if np.all(dirt_map == 0): break  # Done when clean

     env.end()
     ```
     This implements rule-based deduction; adjust map_matrix for dirt positions to see logic in action.

3. **Practical Reasoning Agents**:
   - Principle: Deliberate on desires via beliefs, commit to intentions, plan means-ends.
   - BDI: Update beliefs, generate desires, filter intentions, plan/execute.
   - MAPE-K: Monitor-analyze-plan-execute loop.
   - Example: Air traffic agent plans routes. In ir-sim, simulate BDI for path planning in dynamic environment.
     ```python
     import irsim
     import numpy as np

     # YAML: bdi_world.yaml (multi-robot with goals)
     # world: {height: 10, width: 10}
     # robots: [{kinematics: 'diff', state: [1,1,0], goal: [9,9,0]}, {kinematics: 'diff', state: [9,9,0], goal: [1,1,0]}]

     env = irsim.make('bdi_world.yaml')

     class BDI:
         def __init__(self, goal):
             self.beliefs = {}  # e.g., positions
             self.desires = [goal]  # Options
             self.intention = goal  # Commit
             self.plan = []  # Action sequence

         def update_beliefs(self, state, others):
             self.beliefs['self'] = state
             self.beliefs['others'] = others

         def generate_desires(self):
             # Add avoidance if crowded
             if len(self.beliefs['others']) > 0:
                 self.desires.append('avoid')

         def filter_intentions(self):
             # Prioritize avoidance if needed
             if 'avoid' in self.desires:
                 self.intention = 'avoid'
             else:
                 self.intention = self.desires[0]

         def plan(self):
             if self.intention == 'avoid':
                 self.plan = [np.array([0, np.pi / 4])]  # Turn
             else:
                 dx = self.intention[0] - self.beliefs['self'][0]
                 self.plan = [np.array([0.5, 0 if dx > 0 else np.pi])]  # Toward goal

         def execute(self):
             return self.plan.pop(0) if self.plan else np.array([0, 0])

     bdi_agent = BDI(env.get_goal(0))

     for i in range(500):
         state = env.get_state(0)
         others = [env.get_state(j) for j in range(1, env.robot_num)]
         bdi_agent.update_beliefs(state, others)
         bdi_agent.generate_desires()
         bdi_agent.filter_intentions()
         bdi_agent.plan()
         vel = bdi_agent.execute()
         env.step(vel)
         env.render(0.05)
         if env.done(): break

     env.end()
     ```
     This models BDI loop; add more robots in YAML to trigger avoidance intention.

4. **Self-Aware Agents**:
   - Principle: Build runtime models for prediction and adaptation across awareness levels (stimulus to meta-self).
   - Example: E-health agent adapts plans. In ir-sim, add self-modeling for battery/environment awareness.
     ```python
     import irsim
     import numpy as np

     # YAML: self_aware.yaml (basic robot)
     # robot: {kinematics: 'diff', state: [1,1,0], goal: [9,9,0]}

     env = irsim.make('self_aware.yaml')

     class SelfAware:
         def __init__(self):
             self.model = {'battery': 100, 'env_map': np.zeros((10,10))}  # Self/environment model
             self.awareness_level = 'stimulus'  # Start basic

         def learn_model(self, percepts):
             # Update model (e.g., map obstacles)
             self.model['env_map'] += percepts  # Hypothetical update
             if np.sum(self.model['env_map']) > 10:
                 self.awareness_level = 'goal'  # Advance

         def reason_act(self):
             if self.awareness_level == 'stimulus':
                 return np.random.uniform(-1,1,2)  # Random react
             else:
                 # Plan based on model
                 return np.array([0.5, 0])  # Toward goal

     agent = SelfAware()

     for i in range(300):
         percepts = env.get_lidar_scan(0)  # Stimuli
         agent.learn_model(np.array(percepts) < 2.0)  # Detect nearby
         vel = agent.reason_act()
         env.step(vel)
         agent.model['battery'] -= 1  # Self-update
         env.render(0.05)
         if agent.model['battery'] <= 0: break

     env.end()
     ```
     This builds models incrementally; observe level changes with simulated percepts.

5. **Hybrid Agents**:
   - Principle: Layer reactive and deliberative components horizontally (parallel) or vertically (sequential).
   - Example: Cloud allocator. In ir-sim, hybridize reactive avoidance with planned navigation.
     ```python
     import irsim

     # YAML: hybrid_world.yaml (with obstacles)
     # robot: {kinematics: 'diff', state: [1,1,0], goal: [9,9,0], behavior: 'dash'}

     env = irsim.make('hybrid_world.yaml')

     def hybrid_control(percepts):
         # Reactive layer (horizontal/parallel)
         if any(d < 1 for d in percepts):
             return 'avoid', np.array([0, np.pi/4])
         # Deliberative layer (vertical/sequential)
         else:
             return 'plan', env.cal_des_vel()  # Dash to goal

     for i in range(400):
         percepts = env.get_lidar_scan(0)
         layer, vel = hybrid_control(percepts)
         env.step(vel)
         env.render(0.05)
         if env.done(): break

     env.end()
     ```
     This combines layers; test vertical by sequencing plan after avoid.

#### Real-World Applications
BDI for mission control; subsumption for robotics.

#### Limitations
Reactive: Hard to debug emergence. Deliberative: Slow in uncertainty. Hybrids: Complex integration.

### Short Sum-Up
Architectures enable MAS autonomy via modular designs. Use ir-sim codes to simulate and learn: tweak YAML for environments, experiment with logic for behaviors.
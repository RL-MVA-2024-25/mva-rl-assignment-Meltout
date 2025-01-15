import gymnasium as gym
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from pickle import dump, load
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env_fixed = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
) 

class FittedQIteration:
  def __init__(self, dim_state=6, n_actions=4, gamma=.98):
    self.gamma = gamma
    self.n_actions = 4
    self.dim_state = dim_state
    self.Qfunctions = []
    self.samples = []
    self._sa_buffer = np.zeros((n_actions, dim_state + 1))
    for a in range(n_actions):
      self._sa_buffer[a, dim_state] = a
  
  def save_last_Qfunction(self, path="models/fitted_Q_iteration.pkl"):
    if len(self.Qfunctions) == 0:
      print("Error: Qfunctions list is empty")
      return
    with open(path, "wb") as f:
      dump(self.Qfunctions[-1], f, protocol=5)

  def load_Qfunction(self, path="models/fitted_Q_iteration.pkl"):
    with open(path, "rb") as f:
      Q = load(f)
      self.Qfunctions = [Q]

  def greedy_action(self, state:np.ndarray):
    s = self._augmented_state(state)
    if len(self.Qfunctions) == 0:
      return np.random.randint(4)
    else:
      Q = self.Qfunctions[-1]
      for a in range(self.n_actions):
        self._sa_buffer[a, :len(s)] = s
      return np.argmax(Q.predict(self._sa_buffer))
  
  def get_state_deltas(self, state, action, state2):
    if state is None:
      return [0, 0, 0, 0]
    T1 = state[0]
    T2 = state[2]
    V = state[4]
    T1dot = state2[0]
    T2dot = state2[2]
    Vdot = state2[4]
    return [T1dot - T1, T2dot - T2, Vdot - V, action]


  def collect_samples_eps_greedy(self, env:gym.Env, n_traj_unhealthy=20, n_traj_healthy=0, n_traj_uninfected=0, eps=.15, disable_tqdm=False):
    print("Collecting samples...")
    for mode, n_traj in zip(['unhealthy', 'healthy', 'uninfected'], [n_traj_unhealthy, n_traj_healthy, n_traj_uninfected]):
      for _ in tqdm(range(n_traj), disable=disable_tqdm):
        timestep = 0
        s, _ = env.reset(options={'mode':mode})
        s_deltas = self.get_state_deltas(None, None, s)
        while True:
          if np.random.rand() < eps:
            a = env.action_space.sample()
          else:
            a = self.greedy_action(s)
          s2, r, done, trunc, _ = env.step(a)
          s2_deltas = self.get_state_deltas(s, a, s2)
          self.samples.append((s, a, r, s2, done or trunc, timestep))
          if done or trunc:
            break
          else:
            s = s2
            s_deltas = s2_deltas
            timestep += 1
    

  def monte_carlo_eval(self, env:gym.Env, nb_simulations=1):
    rewards = []
    for _ in range(nb_simulations):
      s, _ = env.reset()
      episode_reward = 0
      while True:
        a = self.greedy_action(s)
        s2, r, dead, trunc, _ = env.step(a)
        episode_reward += r
        if dead or trunc:
          break
        else:
          s = s2
      rewards.append(episode_reward)
    return np.mean(rewards)

  
  def _print_progress(self, env:gym.Env, mc_eval_score_fixed=None, mc_eval_score_randomized=None):
    s0, _ = env.reset()
    Qs0a = []
    for a in range(self.n_actions):
      Qs0a.append(self.Qfunctions[-1].predict(np.append(self._augmented_state(s0), a).reshape(1, -1)))
    
    print(f"Predicted reward in initial state of unhealthy patient: {np.max(Qs0a)}")
    if mc_eval_score_fixed is not None:
      print(f"Mean reward obtained from Monte Carlo FIXED environment: {mc_eval_score_fixed:.7E}")
    if mc_eval_score_randomized is not None:
      print(f"Mean reward obtained from Monte Carlo RANDOMIZED environment: {mc_eval_score_randomized:.7E}")

  def _augmented_state(self, state):
    return state
    T1 = state[0]
    T1star = state[1]
    T2 = state[2]
    T2star = state[3]
    V = state[4]
    E = state[5]
    T1_ratio = T1 / (T1 + T1star)
    T2_ratio = T2 / (T2 + T2star)
    return np.append(state, [T1_ratio, T2_ratio])

  def _process_samples(self):
    S = np.array([self._augmented_state(s) for (s, _, _, _, _, _) in self.samples])
    A = np.array([a for (_, a, _, _, _, _) in self.samples]).reshape(-1, 1)
    R = np.array([r for (_, _, r, _, _, _) in self.samples])
    S2 = np.array([self._augmented_state(s2) for (_, _, _, s2, _, _) in self.samples])
    D = np.array([d for (_, _, _, _, d, _) in self.samples], dtype=int)
    T = np.array([t for (_, _, _, _, _, t) in self.samples])

    return S, A, R, S2, D, T
  
  def train(self, env:gym.Env, num_epochs=100, iterations_per_epoch=200):
    best_score_fixed = 0
    best_score_random = 0
    best_Q_function = None if len(self.Qfunctions) == 0 else self.Qfunctions[0]
    new_best_function_found = (best_Q_function is not None)
    eps_schedule = np.linspace(0.25, 0.02, num=num_epochs)
    for epoch, eps in zip(range(num_epochs), eps_schedule):
      # append best_Q_function to self.Qfunctions in order to collect samples by using it
      if best_Q_function is not None:
        self.Qfunctions.append(best_Q_function)
      # if new_best_function_found:
      #   self.collect_samples_eps_greedy(env, eps=eps)
      # else:
      #   self.collect_samples_eps_greedy(env, eps=0.8)
      self.collect_samples_eps_greedy(env, eps=eps)
      new_best_function_found = False
      S, A, R, S2, D, T = self._process_samples()
      print(S.shape, A.shape, R.shape, S2.shape, D.shape, T.shape)
      SA = np.append(S, A, axis=1)
      print(f"Starting epoch {epoch + 1}...")
      for i in tqdm(range(iterations_per_epoch)):
        if len(self.Qfunctions) == 0:
          target = R.copy()
        else:
          Q2 = np.empty(shape=(self.n_actions, len(self.samples)))
          for a2 in range(self.n_actions):
            S2A2 = np.append(S2, np.ones(shape=(len(self.samples), 1)), axis=1)
            Q2[a2] = self.Qfunctions[-1].predict(S2A2)
          target = R + self.gamma*(1-D)*np.max(Q2, axis=0)
          
        Q = xgb.XGBRegressor(n_estimators=50)
        Q.fit(SA, target)
        self.Qfunctions.append(Q)
        
        mc_eval_score_fixed = self.monte_carlo_eval(env_fixed, nb_simulations=1)
        if mc_eval_score_fixed > best_score_fixed:
          best_score_fixed = mc_eval_score_fixed
          green_col = '\033[92m'
          endcol = '\033[0m'
          print(f"{green_col}New best FIXED score: {best_score_fixed:.7E}{endcol}")
          self.save_last_Qfunction("models/best_fixed_patient_model.pkl")
          if best_Q_function is None:
            best_Q_function = self.Qfunctions[-1]
            new_best_function_found = True
          
        if mc_eval_score_fixed > 2e10:
          mc_eval_score_random = self.monte_carlo_eval(env, nb_simulations=50)
          if mc_eval_score_random > best_score_random:
            best_score_random = mc_eval_score_random
            self.save_last_Qfunction()
            best_score_random = mc_eval_score_random
            green_col = '\033[92m'
            endcol = '\033[0m'
            print(f"{green_col}New best RANDOM score: {best_score_random:.7E}{endcol}")
            best_Q_function = self.Qfunctions[-1]
            new_best_function_found = True

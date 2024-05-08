# SARSA Learning Algorithm

## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

## SARSA LEARNING ALGORITHM
### Step 1:
Initialize the Q-table with random values for all state-action pairs.
### Step 2:
Initialize the current state S and choose the initial action A using an epsilon-greedy policy based on the Q-values in the Q-table.
### Step 3:
Repeat until the episode ends and then take action A and observe the next state S' and the reward R.
### Step 4:
Update the Q-value for the current state-action pair (S, A) using the SARSA update rule.
### Step 5:
Update State and Action and repeat the step 3 untill the episodes ends.


## SARSA LEARNING FUNCTION
```py
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state,Q,epsilon: 
    			np.argmax(Q[state]) 
    			if np.random.random() > epsilon 
                else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)

    for e in tqdm(range(n_episodes),leave=False):
        state, done = env.reset(), False
        action = select_action(state,Q,epsilons[e])

        while not done:
            next_state,reward,done,_ = env.step(action)
            next_action = select_action(next_state,Q,epsilons[e])

            td_target = reward+gamma*Q[next_state][next_action]*(not done)

            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state, action = next_state,next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q,axis=1))

    V = np.max(Q,axis=1)
    pi = lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
### Optimal State Value Functions:
![280444934-3dae6210-e070-4b40-9b2e-e357bca31feb](https://github.com/Manojrathinavelu/sarsa-learning/assets/119560395/8574b25e-904a-4ac6-bc93-a31d483982aa)

### Optimal Action Value Functions:
![280444985-7877b5f8-a7d2-44f5-a3c6-c822ca282043](https://github.com/Manojrathinavelu/sarsa-learning/assets/119560395/262b1ab7-3879-4d78-8f97-76fb0081eb31)


### First Visit Monte Carlo Estimates
![280445013-695e052e-36a4-4634-b301-3ca4f91da419](https://github.com/Manojrathinavelu/sarsa-learning/assets/119560395/667918ce-2457-4d9e-aca3-81899f0b4e6e)

### Sarsa Estimates:
![280445027-b5dcb64d-1395-4a66-ad6e-bd498375e22b](https://github.com/Manojrathinavelu/sarsa-learning/assets/119560395/f89a5936-6f1c-4d29-a793-36040a2c9ad8)


## RESULT:
Thus the optimal policy for the given RL environment is found using SARSA-Learning and the state values are compared with the Monte Carlo method.

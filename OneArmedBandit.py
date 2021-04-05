import numpy as np
import matplotlib.pyplot as plt

def generate_weight():
    r = np.random.randint(1,11)
    m = r**1.5
    return m

def reward_scrap(m):
    return 0.7*m

def monte_carlo(generator_func, eval_func, N = 1000):
    total = 0
    evaluations = []
    for _ in range(N):
        sample = generator_func()
        value = eval_func(sample)
        evaluations.append(value)
        total += value
    mean = total/N
    std = (sum(((v-mean)**2 for v in evaluations))/N)**0.5
    return (mean, std)

def error_estimate(mean, std, N):
    error = std/(N**0.5)
    return (mean - error, mean + error)

def present(mean, std, error):
    lower, upper = error
    print('mean: ' + str(mean))
    print('std: ' + str(std))
    print('error bound: ' + str(lower) + ' - ' + str(upper))
    return 


def mc_simulate_and_present(reward_func, N = 1000):
    mean, std = monte_carlo(generate_weight, reward_func, N)
    error = error_estimate(mean, std, N)
    present(mean, std, error)

mc_simulate_and_present(reward_scrap, 1000)


def reward_hang(m):
    return 100 + 1.1 * (m**0.5)

mc_simulate_and_present(reward_hang, 1000)

def prob_break_given_params(m, k, lam):
    return 1 - np.math.e**-((m/(1+abs(lam)))**(1+abs(k)))
prob_break_given_params(5, 10, 4)

def simulate_prob_break(m, mean_k, std_k, mean_lam, std_lam):
    k = np.random.normal(mean_k, std_k)
    lam = np.random.normal(mean_lam, std_lam)
    return prob_break_given_params(m, k, lam)

# Paremeters for brands
brands = [
    [9,1,9,3,10],
    [7,2,11,2,15],
    [15,3,4,2,5],
    [12,1,4,1,15],
    [15,5,15,5,15],
    [14,3,8,2,10],
    [9,1,9,4,10],
    [6,3,7,3,5],
    [13,1,12,1,15],
    [10,2,6,3,5]
]

# Simulated reward given a brand
def reward_brand(m, brand):
    params = brands[brand]
    mean_k = params[0]
    std_k = params[1]
    mean_lam = params[2]
    std_lam = params[3]
    price = params[4]

    sim_prob_break = simulate_prob_break(m, mean_k, std_k, mean_lam, std_lam)
    
    expected_reward = sim_prob_break * reward_scrap(m) + (1-sim_prob_break) * reward_hang(m) - price
    
    return expected_reward

# Returns a reward function that only needs 'm' as parameter
def reward_func_given_brand(brand):
    return lambda m: reward_brand(m, brand)

def level_3():
    expectd_earnings = []
    for b in range(len(brands)):
        print('brand: ' + str(b+1))
        reward_func = reward_func_given_brand(b)
        mean, _ = monte_carlo(generate_weight, reward_func)
        expectd_earnings.append(mean)
        print(mean)
        print()
    
    
    best_brand = np.argmax(expectd_earnings)
    best_earnings = expectd_earnings[best_brand]
    
    print('best brand: ' + str(best_brand+1) + ' - ' + str(best_earnings))
    print()

    average_earnings = sum(expectd_earnings)/len(expectd_earnings)
    print('expected earnings random brands: ' + str(average_earnings))
level_3()

def reward_func_e_gready(brand):
    m = generate_weight()
    return reward_brand(m, brand)

def e_gready(e, num_actions = 10, reward_func = reward_func_e_gready, a = 0.1, itr = 20, initial_avs = None):
    action_values = []
    if initial_avs == None:
        action_values = [0]*num_actions
    else:
        action_values = initial_avs
    
    rewards = []
    actions = []
    action = 0
    for _ in range(itr):
        r = np.random.rand()
        if r < e:
            action = np.random.randint(0, num_actions)
        else:
            action = np.argmax(action_values)
        
        reward = reward_func(action)
        action_values[action] += a*(reward - action_values[action])
        rewards.append(reward)
        actions.append(action)

    return rewards, actions


def multible_episodes(e, num_episodes=20):
    episodes = []
    for _ in range(num_episodes):
        episodes.append(e_gready(e)[0])
    return episodes
    
def plot_average_earnings_per_BT(e, num_episodes=20):
    episodes = multible_episodes(e, num_episodes)
    episodes = np.matrix(episodes)
    average_BT = np.mean(episodes, axis=0)
    
    y = average_BT[-1]
    x = np.matrix(range(len(y)))

    plt.plot(x,y)


plot_average_earnings_per_BT(0.1)

def plot_total_earnings(es):
    es_episodes = [multible_episodes(e) for e in es]
    es_total_earnings_per_episode = [[sum(episode) for episode in e] for e in es_episodes]
    es_average_earnings = [sum(episodes)/len(episodes) 
        for episodes in es_total_earnings_per_episode]

    x = es
    y = es_average_earnings

    plt.plot(x,y)

optimal_e = 0.1
episodes = []
num_episodes = 100
for i in range(num_episodes):
    for _ in range(num_episodes):
        episodes.append(e_gready(optimal_e))
num_actions = [0]*10
episodes = np.matrix(episodes)
for episode in episodes:
    actions = episode[:,1]
    for a in actions:
        num_actions[a] += 1
average_action = [num_action / num_episodes for num_action in num_actions]
print(average_action)





 

    






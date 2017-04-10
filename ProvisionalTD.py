import numpy as np
import gym
import functools
import sys,os,shutil,csv


LOG_PATH = './rl_logs/'
LG = ''
MAX_EXPERIMENTS = 40

LEARNING_RATE = 0.01


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_experiment():
    print("setting up")
    global EXPERIMENT_NO
    ensure_dir(LOG_PATH)
    experiments = [i for i in os.listdir(LOG_PATH) if "experiment" in i]
    for i in experiments:
        try:
            os.rmdir(LOG_PATH + i)
        except (OSError, NotADirectoryError):
            continue

    get_num = lambda x: int(''.join(ele for ele in x if ele.isdigit()))
    exps = sorted([i for i in os.listdir(LOG_PATH) if "experiment" in i], reverse=True,
        key=get_num)

    if len(exps) >= MAX_EXPERIMENTS:
        print("Removing Experiment {}".format(len(exps) - 1))
        shutil.rmtree(LOG_PATH + exps[-1])
        exps.pop(-1)

    for i,el in enumerate(exps):
        os.rename(LOG_PATH + el, LOG_PATH + "experiment_" + str(len(exps) - i))

    EXPERIMENT_NO = 0
    print("writing to " + LOG_PATH + "experiment_" + str(EXPERIMENT_NO))
    ensure_dir(LOG_PATH + "experiment_" + str(EXPERIMENT_NO))
    return LOG_PATH + "experiment_" + str(EXPERIMENT_NO) + "/"

def monte_carlo_delta(traj, V, gamma, lamb):
    rho_k = traj[0][-1]

    states = [i[0] for i in traj]
    rhos = [i[-1] for i in traj]
    rs   = [i[2] for i in traj]
    delt = 0
    t = len(traj)

    for i in range(1, t):
        C = functools.reduce(lambda x,y: x * y * gamma * lamb, rhos[1:i], 1)
        r_sum = functools.reduce(lambda x,y: x + y,rs[0:i])
        eps = r_sum - states[0].dot(V)
        d   = r_sum - states[0].dot(V) + states[i].dot(V)
        delt += C * ((1 - gamma) * eps + gamma * (1 - lamb) * d)

    C = functools.reduce(lambda x,y: x * y * gamma * lamb, rhos[1:], 1)
    r_sum = functools.reduce(lambda x,y: x + y,rs)
    eps = r_sum - states[0].dot(V)
    d   = r_sum - states[0].dot(V) + traj[-1][1].dot(V)
    delt = rho_k * delt + rho_k * C * ((1 - gamma) * eps + gamma * d)
    return delt

def learn(behaviour_policy, target_policy, env, n_episodes=10000, alpha=LEARNING_RATE, gamma=0.9, lamb=1):

    LG = setup_experiment()
    V = np.zeros(env.shape[0] * env.shape[1])
    
    for episode in range(n_episodes):
       
        
        is_terminal = False
        s = env.reset()
        n_steps = 0

        e = np.zeros(V.shape)
        u = np.zeros(V.shape)

        remember_deltas = []
        remember_sar    = []

        # pre-episode logging
        ep_reward = 0

        while not is_terminal and n_steps < 200:
            a_probs = s.dot(behaviour_policy)
            a  = np.random.choice(len(a_probs), p=a_probs)

            rho = s.dot(target_policy)[a] / a_probs[a]
            
            s_,r,is_terminal,other = env.step(a)

            # Trace Change
            e = rho * (s + gamma * lamb * e)

            # Value Change
            delta = r - s.dot(V)
            if not is_terminal:
                delta += gamma * s_.dot(V)
            remember_deltas.append(alpha * delta * e + (rho - 1) * u)

            # Provisional Change
            u = gamma * lamb * (rho * u + alpha * (s_.dot(V) - s.dot(V) + r) * e)

            remember_sar.append((s,s_,r,rho))


            s = s_
            n_steps += 1


            #in-episode logging
            ep_reward += r

            """
            # Computes and logs forward view
            mcd_log = np.zeros(V.shape)
            mcds = []
            for i in range(len(remember_sar)):
                mcd = monte_carlo_delta(remember_sar[i:], V, gamma, lamb)
                mcd_log += alpha * mcd * remember_sar[i][0]
                mcds.append(mcd)

            deltas_log = np.zeros(V.shape)
            for i in remember_deltas:
                deltas_log += i

            with open(LG + "delta_dist.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow([((mcd_log - deltas_log)**2).sum()])
            with open(LG + "delta.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow([delta])
            """




        # end for step in episode

        delta = functools.reduce(lambda x,y: x+y, remember_deltas)

        """
        mag = delta.dot(delta)
        if mag > 20:
            with open(LG + "sas.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                for s,s_,r,rho in remember_sar:
                    writer.writerow([s.argmax(), r, rho])
            with open(LG + "delt_s.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                for i in remember_deltas:
                    writer.writerow([i.dot(i)])
            break
        """


        V = V + delta


        # after-episode logging
        with open(LG + "value.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(V)
        """
        with open(LG + "num_steps.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([n_steps])
        """

        if episode % int(n_episodes / 6) == 0:
            print("Completed Episode {}".format(episode))
        elif episode % 100 == 0:
            print("Completed Episode {}     ".format(episode), end="\r")
            sys.stdout.flush()

    # end for episode
# end learn()

def learn_td(behaviour_policy, target_policy, env, n_episodes=10000, alpha=LEARNING_RATE, gamma=0.9, lamb=1):

    LG = setup_experiment()
    V = np.zeros(env.shape[0] * env.shape[1])
    
    for episode in range(n_episodes):
       
        
        is_terminal = False
        s = env.reset()
        n_steps = 0

        e = np.zeros(V.shape)
        u = np.zeros(V.shape)

        remember_deltas = []
        # remember_sar    = []

        # pre-episode logging
        ep_reward = 0

        while not is_terminal and n_steps < 500:
            a_probs = s.dot(behaviour_policy)
            a  = np.random.choice(len(a_probs), p=a_probs)

            
            s_,r,is_terminal,other = env.step(a)

            # Trace Change
            e = (s + gamma * lamb * e)

            # Value Change
            delta = r - s.dot(V)
            if not is_terminal:
                delta += gamma * s_.dot(V)
            remember_deltas.append(alpha * delta * e)


            s = s_
            n_steps += 1


            #in-episode logging
            ep_reward += r

        # end for step in episode

        V = V + functools.reduce(lambda x,y: x+y, remember_deltas)


        # after-episode logging
        with open(LG + "value.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(V)
        with open(LG + "num_steps.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([n_steps])

        if episode % int(n_episodes / 6) == 0:
            print("Completed Episode {}".format(episode))
        elif episode % 100 == 0:
            print("Completed Episode {}     ".format(episode), end="\r")
            sys.stdout.flush()

    # end for episode
# end learn()

def learn_p_retrace(behaviour_policy, target_policy, env, n_episodes=10000, alpha=LEARNING_RATE, gamma=0.9, lamb=1):

    LG = setup_experiment()
    V = np.zeros(env.shape[0] * env.shape[1])
    
    for episode in range(n_episodes):
       
        
        is_terminal = False
        s = env.reset()
        n_steps = 0

        e = np.zeros(V.shape)
        u = np.zeros(V.shape)

        remember_deltas = []
        # remember_sar    = []

        # pre-episode logging
        ep_reward = 0

        while not is_terminal and n_steps < 500:
            a_probs = s.dot(behaviour_policy)
            a  = np.random.choice(len(a_probs), p=a_probs)

            rho = min(s.dot(target_policy)[a] / a_probs[a], 1)
            
            s_,r,is_terminal,other = env.step(a)

            # Trace Change
            e = rho * (s + gamma * lamb * e)

            # Value Change
            delta = r - s.dot(V)
            if not is_terminal:
                delta += gamma * s_.dot(V)
            remember_deltas.append(alpha * delta * e + (rho - 1) * u)

            # Provisional Change
            u = gamma * lamb * (rho * u + alpha * (s_.dot(V) - s.dot(V) + r) * e)

            # remember_sar.append((s,s_,r,rho))


            s = s_
            n_steps += 1


            #in-episode logging
            ep_reward += r

            """
            # Computes and logs forward view
            mcd_log = np.zeros(V.shape)
            mcds = []
            for i in range(len(remember_sar)):
                mcd = monte_carlo_delta(remember_sar[i:], V, gamma, lamb)
                mcd_log += alpha * mcd * remember_sar[i][0]
                mcds.append(mcd)

            deltas_log = np.zeros(V.shape)
            for i in remember_deltas:
                deltas_log += i

            with open(LG + "delta_dist.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow([((mcd_log - deltas_log)**2).sum()])
            """




        # end for step in episode

        V = V + functools.reduce(lambda x,y: x+y, remember_deltas)


        # after-episode logging
        with open(LG + "value.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(V)
        with open(LG + "num_steps.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([n_steps])

        if episode % int(n_episodes / 6) == 0:
            print("Completed Episode {}".format(episode))
        elif episode % 100 == 0:
            print("Completed Episode {}     ".format(episode), end="\r")
            sys.stdout.flush()

    # end for episode
# end learn()
def learn_rate_limited(behaviour_policy, target_policy, env, n_episodes=10000, alpha=LEARNING_RATE, gamma=0.9, lamb=1):

    LG = setup_experiment()
    V = np.zeros(env.shape[0] * env.shape[1])
    
    for episode in range(n_episodes):
       
        
        is_terminal = False
        s = env.reset()
        n_steps = 0

        e = np.zeros(V.shape)
        u = np.zeros(V.shape)

        remember_deltas = []
        remember_u = []
        remember_sar = []

        # pre-episode logging
        ep_reward = 0

        while not is_terminal and n_steps < 500:
            a_probs = s.dot(behaviour_policy)
            a  = np.random.choice(len(a_probs), p=a_probs)

            rho = s.dot(target_policy)[a] / a_probs[a]
            
            s_,r,is_terminal,other = env.step(a)

            # Trace Change
            e = rho * (s + gamma * lamb * e)

            # Value Change
            delta = r - s.dot(V)
            if not is_terminal:
                delta += gamma * s_.dot(V)
            remember_deltas.append(alpha * delta * e + (min(rho, 1) - 1) * u)

            # Provisional Change
            u = gamma * lamb * (min(rho, 1) * u + alpha * (s_.dot(V) - s.dot(V) + r) * e)

            remember_sar.append((s,s_,r,rho,n_steps))
            remember_u.append(u)


            s = s_
            n_steps += 1


            #in-episode logging
            ep_reward += r




        # end for step in episode
        delta = functools.reduce(lambda x,y: x+y, remember_deltas)

        mag = delta.dot(delta)
        if mag > 20:
            with open(LG + "sas.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                for s,s_,r,rho, is_terminal in remember_sar:
                    writer.writerow([s.argmax(), s_.argmax(), r, rho, is_terminal])
            with open(LG + "delt_s.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                for i in remember_deltas:
                    writer.writerow([i.dot(i)])
            with open(LG + "u_log.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                for i in remember_u:
                    writer.writerow(i)
            break
        V = V + delta 


        # after-episode logging
        with open(LG + "value.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(V)
        with open(LG + "num_steps.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([n_steps])

        if episode % int(n_episodes / 6) == 0:
            print("Completed Episode {}".format(episode))
        elif episode % 100 == 0:
            print("Completed Episode {}     ".format(episode), end="\r")
            sys.stdout.flush()

    # end for episode
# end learn()

def learn_p_rate_limited(behaviour_policy, target_policy, env, n_episodes=10000, alpha=LEARNING_RATE, gamma=0.9, lamb=1):

    LG = setup_experiment()
    V = np.zeros(env.shape[0] * env.shape[1])
    
    for episode in range(n_episodes):
       
        
        is_terminal = False
        s = env.reset()
        n_steps = 0

        e = np.zeros(V.shape)
        u = np.zeros(V.shape)

        remember_deltas = []

        # pre-episode logging
        ep_reward = 0

        while not is_terminal and n_steps < 500:
            a_probs = s.dot(behaviour_policy)
            a  = np.random.choice(len(a_probs), p=a_probs)

            rho = s.dot(target_policy)[a] / a_probs[a]
            
            s_,r,is_terminal,other = env.step(a)

            # Trace Change
            e = rho * (s + gamma * lamb * e)

            # Value Change
            delta = r - s.dot(V)
            if not is_terminal:
                delta += gamma * s_.dot(V)
            remember_deltas.append(alpha * delta * e + (rho - 1) * u)

            # Provisional Change
            u = gamma * lamb * (rho * u + alpha * (s_.dot(V) - s.dot(V) + r) * e)



            s = s_
            n_steps += 1


            #in-episode logging
            ep_reward += r




        # end for step in episode
        delta = functools.reduce(lambda x,y: x+y, remember_deltas)
        dd = np.copy(delta)
        d_sign = (delta > 0) * 1 - (delta < 0) * 1
        a = np.abs(delta)
        b = np.ones(delta.shape)
        delta = np.minimum(a,b * 0.1)
        delta = delta * d_sign


        V = V + delta 


        # after-episode logging
        with open(LG + "value.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(V)
        with open(LG + "num_steps.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([n_steps])

        if episode % int(n_episodes / 6) == 0:
            print("Completed Episode {}".format(episode))
        elif episode % 100 == 0:
            print("Completed Episode {}     ".format(episode), end="\r")
            sys.stdout.flush()

    # end for episode
# end learn()

def learn_retrace(behaviour_policy, target_policy, env, n_episodes=10000, alpha=LEARNING_RATE, gamma=0.9, lamb=1):

    LG = setup_experiment()
    V = np.zeros(env.shape[0] * env.shape[1])
    
    for episode in range(n_episodes):
       
        
        is_terminal = False
        s = env.reset()
        n_steps = 0

        e = np.zeros(V.shape)
        u = np.zeros(V.shape)

        remember_deltas = []
        # remember_sar    = []

        # pre-episode logging
        ep_reward = 0

        while not is_terminal and n_steps < 500:
            a_probs = s.dot(behaviour_policy)
            a  = np.random.choice(len(a_probs), p=a_probs)

            rho = min(s.dot(target_policy)[a] / a_probs[a], 1)
            
            s_,r,is_terminal,other = env.step(a)

            # Trace Change
            e = rho * (s + gamma * lamb * e)

            # Value Change
            delta = r - s.dot(V)
            if not is_terminal:
                delta += gamma * s_.dot(V)
            remember_deltas.append(alpha * delta * e)

            s = s_
            n_steps += 1


            #in-episode logging
            ep_reward += r

            """
            # Computes and logs forward view
            mcd_log = np.zeros(V.shape)
            mcds = []
            for i in range(len(remember_sar)):
                mcd = monte_carlo_delta(remember_sar[i:], V, gamma, lamb)
                mcd_log += alpha * mcd * remember_sar[i][0]
                mcds.append(mcd)

            deltas_log = np.zeros(V.shape)
            for i in remember_deltas:
                deltas_log += i

            with open(LG + "delta_dist.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow([((mcd_log - deltas_log)**2).sum()])
            """




        # end for step in episode

        V = V + functools.reduce(lambda x,y: x+y, remember_deltas)


        # after-episode logging
        with open(LG + "value.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(V)
        with open(LG + "num_steps.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([n_steps])

        if episode % int(n_episodes / 6) == 0:
            print("Completed Episode {}".format(episode))
        elif episode % 100 == 0:
            print("Completed Episode {}     ".format(episode), end="\r")
            sys.stdout.flush()

    # end for episode
# end learn()

def learn_is(behaviour_policy, target_policy, env, n_episodes=10000, alpha=LEARNING_RATE, gamma=0.9, lamb=1):

    LG = setup_experiment()
    V = np.zeros(env.shape[0] * env.shape[1])
    
    for episode in range(n_episodes):
       
        
        is_terminal = False
        s = env.reset()
        n_steps = 0

        e = np.zeros(V.shape)
        u = np.zeros(V.shape)

        remember_deltas = []
        # remember_sar    = []

        # pre-episode logging
        ep_reward = 0

        while not is_terminal and n_steps < 500:
            a_probs = s.dot(behaviour_policy)
            a  = np.random.choice(len(a_probs), p=a_probs)

            rho = s.dot(target_policy)[a] / a_probs[a]
            
            s_,r,is_terminal,other = env.step(a)

            # Trace Change
            e = rho * (s + gamma * lamb * e)

            # Value Change
            delta = r - s.dot(V)
            if not is_terminal:
                delta += gamma * s_.dot(V)
            remember_deltas.append(alpha * delta * e)

            s = s_
            n_steps += 1


            #in-episode logging
            ep_reward += r

            """
            # Computes and logs forward view
            mcd_log = np.zeros(V.shape)
            mcds = []
            for i in range(len(remember_sar)):
                mcd = monte_carlo_delta(remember_sar[i:], V, gamma, lamb)
                mcd_log += alpha * mcd * remember_sar[i][0]
                mcds.append(mcd)

            deltas_log = np.zeros(V.shape)
            for i in remember_deltas:
                deltas_log += i

            with open(LG + "delta_dist.csv", 'a+') as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow([((mcd_log - deltas_log)**2).sum()])
            """




        # end for step in episode

        V = V + functools.reduce(lambda x,y: x+y, remember_deltas)


        # after-episode logging
        with open(LG + "value.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(V)
        with open(LG + "num_steps.csv", 'a+') as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([n_steps])

        if episode % int(n_episodes / 6) == 0:
            print("Completed Episode {}".format(episode))
        elif episode % 100 == 0:
            print("Completed Episode {}     ".format(episode), end="\r")
            sys.stdout.flush()

    # end for episode
# end learn()


if __name__ == '__main__':
    pass
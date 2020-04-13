import numpy as np


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def vec_rollout(
        env,
        n_envs, #added this to specify # environments
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = [[] for _ in  range(n_envs)]
    actions = [[] for _ in range(n_envs)]
    rewards = [[] for _ in  range(n_envs)]
    terminals = [[] for _ in  range(n_envs)]
    agent_infos = [[] for _ in  range(n_envs)]
    env_infos = [[] for _ in  range(n_envs)]
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)

    
    #keep track of each env. 
    env_idx = list(range(n_envs))
    last = {}
    while path_length < max_path_length:
        # print(path_length)
        tmp_action = []
        action = []
        agent_info = []
        #Have to step through all no matter what. Only keep if in env_idx
        for i in range(n_envs):
            a, ai = agent.get_action(o[i])
            action.append(a)
            agent_info.append(ai)
        next_o, r, d, env_info = env.step(action)
        for i in env_idx:
            observations[i].append(o[i])
            actions[i].append(action[i])
            rewards[i].append(r[i])
            terminals[i].append(d[i])
            env_infos[i].append(env_info[i])
            agent_infos[i].append(agent_info[i])

        path_length += 1
        n = len(env_idx)
        #check if any are finished. If so, ignore them. 
        tmp = []
        for i in range(n):
            if not d[env_idx[i]]:
                tmp.append(env_idx[i])
            else:
                last[i] = next_o[i]
        env_idx = tmp
        if len(env_idx) == 0:
            break

        o = next_o
        if render:
            env.render(**render_kwargs)
    for k, v in last.items():
        next_o[k] = v 
    # observations = list(np.concatenate(observations))
    # actions = list(np.concatenate(actions))
    # rewards = list(np.concatenate(rewards))
    # terminals = list(np.concatenate(terminals))
    # env_info = list(np.concatenate(env_info))
    # agent_info = list(np.concatenate(agent_info))

    # actions = np.array(actions)
    # if len(actions.shape) == 1:
    #     actions = np.expand_dims(actions, 1)
    # observations = np.array(observations)
    # if len(observations.shape) == 1:
    #     observations = np.expand_dims(observations, 1)
    #     next_o = np.array([next_o])
    #need to return each run of observations for each env on its own
    ans = []
    for i in range(n_envs):
        observations[i] = np.array(observations[i])
        if len(observations[i].shape) == 1:    
            observations[i] = np.expand_dims(observations[i], 1)
            next_o[i] = np.array([next_o[i]])
        actions[i] = np.array(actions[i])
        if len(actions[i].shape) == 1:
            actions[i] = np.expand_dims(actions[i], 1)
        #last transition should be same. 
        next_observations = np.vstack(
            (
                observations[i][1:, :],
                np.expand_dims(next_o[i], 0)
            )
        )
        item = dict(
            observations=observations[i],
            actions=actions[i],
            rewards=np.array(rewards[i]).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals[i]).reshape(-1, 1),
            agent_infos=agent_infos[i],
            env_infos=env_infos[i],
        )
        ans.append(item)
    return ans

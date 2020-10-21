import torch as th


def build_td_lambda_targets__old(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    bs = rewards.size(0)
    max_t = rewards.size(1)
    targets = rewards.new(target_qs.size()).zero_()[:,:-1] # Produce 1 less target than the inputted Q-Values
    running_target = rewards.new(bs, n_agents).zero_()
    terminated = terminated.float()
    for t in reversed(range(max_t)):
        if t == max_t - 1:
            running_target = mask[:, t] * (rewards[:, t] + gamma * (1 - terminated[:, t]) * target_qs[:, t])
        else:
            running_target = mask[:, t] * (
                    terminated[:, t] * rewards[:, t]  # Just the reward if the env terminates
                 + (1 - terminated[:, t]) * (rewards[:, t] + gamma * (td_lambda * running_target + (1 - td_lambda) * target_qs[:, t]))
            )
        targets[:, t, :] = running_target
    return targets


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


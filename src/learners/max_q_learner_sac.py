import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
from collections import deque
from torch.distributions import Categorical
from controllers import REGISTRY as mac_REGISTRY

class SACQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.mac_params = list(mac.parameters())
        self.params = list(self.mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        assert args.mixer is not None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.mixer_params = list(self.mixer.parameters())
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # Central Q
        # TODO: Clean this mess up!
        self.central_mac = None
        assert self.args.central_mixer == "ff"
        self.central_mixer = QMixerCentralFF(args)
        assert args.central_mac == "basic_central_mac"
        self.central_mac = mac_REGISTRY[args.central_mac](scheme, args) # Groups aren't used in the CentralBasicController. Little hacky
        self.target_central_mac = copy.deepcopy(self.central_mac)
        self.params += list(self.central_mac.parameters())
        self.params += list(self.central_mixer.parameters())
        self.target_central_mixer = copy.deepcopy(self.central_mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Current policies
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Mask out unavailable actions, renormalise (as in action selection)
        mac_out[avail_actions == 0] = 0
        mac_out = mac_out/mac_out.sum(dim=-1, keepdim=True)
        mac_out[avail_actions == 0] = 0
        mac_out[(mac_out.sum(dim=-1, keepdim=True) == 0).expand_as(mac_out)] = 1 # Set any all 0 probability vectors to all 1s. They will be masked out later, but still need to be sampled.

        # Target policies
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions, renormalise (as in action selection)
        target_mac_out[avail_actions == 0] = 0
        target_mac_out = target_mac_out/target_mac_out.sum(dim=-1, keepdim=True)
        target_mac_out[avail_actions == 0] = 0
        target_mac_out[(target_mac_out.sum(dim=-1, keepdim=True) == 0).expand_as(target_mac_out)] = 1 # Set any all 0 probability vectors to all 1s. They will be masked out later, but still need to be sampled.

        # Sample actions
        sampled_actions = Categorical(mac_out).sample().long()
        sampled_target_actions = Categorical(target_mac_out).sample().long()

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        # Actions chosen from replay buffer
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        central_target_action_qvals_agents = th.gather(central_target_mac_out[:,:], 3, \
                                                       sampled_target_actions[:,:].unsqueeze(3).unsqueeze(4)\
                                                        .repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
        # ---

        critic_bootstrap_qvals = self.target_central_mixer(central_target_action_qvals_agents[:,1:], batch["state"][:,1:])

        target_chosen_action_probs = th.gather(target_mac_out, dim=3, index=sampled_target_actions.unsqueeze(3)).squeeze(dim=3)
        target_policy_logs = th.log(target_chosen_action_probs).sum(dim=2, keepdim=True) # Sum across agents
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * \
                  (critic_bootstrap_qvals - self.args.entropy_temp * target_policy_logs[:,1:])

        # Training Critic
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        # Actor Loss
        central_sampled_action_qvals_agents = th.gather(central_mac_out[:, :-1], 3, \
                                                        sampled_actions[:, :-1].unsqueeze(3).unsqueeze(4) \
                                                        .repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)
        central_sampled_action_qvals = self.central_mixer(central_sampled_action_qvals_agents, batch["state"][:,:-1]).repeat(1,1,self.args.n_agents)
        sampled_action_probs = th.gather(mac_out, dim=3, index=sampled_actions.unsqueeze(3)).squeeze(3)
        policy_logs = th.log(sampled_action_probs)[:,:-1]
        actor_mask = mask.expand_as(policy_logs)
        actor_loss = ((policy_logs * (self.args.entropy_temp * (policy_logs + 1) - central_sampled_action_qvals).detach()) * actor_mask).sum()/actor_mask.sum()

        loss = self.args.actor_loss * actor_loss + self.args.central_loss * central_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.grad_norm = grad_norm

        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            ps = mac_out[:, :-1] * avail_actions[:, :-1]
            log_ps = th.log(mac_out[:, :-1] + 0.00001) * avail_actions[:, :-1]
            actor_entropy = -(((ps * log_ps).sum(dim=3) * mask).sum() / mask.sum())
            self.logger.log_stat("actor_entropy", actor_entropy.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        if self.central_mac is not None:
            self.target_central_mac.load_state(self.central_mac)
        self.target_central_mixer.load_state_dict(self.central_mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.central_mac is not None:
            self.central_mac.cuda()
            self.target_central_mac.cuda()
        self.central_mixer.cuda()
        self.target_central_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
from collections import deque
from controllers import REGISTRY as mac_REGISTRY

class DDPGQLearner:
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

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_agents = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals = chosen_action_qvals_agents

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time

        # Max over target Q-Values
        if self.args.double_q:
            raise Exception("No double q for DDPG")
        else:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            _, cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)
            target_mac_out_detach = target_mac_out.clone().detach()
            target_mac_out_detach[avail_actions == 0] = -9999999
            _, tar_max_actions = target_mac_out_detach[:, :].max(dim=3, keepdim=True)

        # Central MAC stuff
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)  # Remove the last dim

        central_target_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            central_target_mac_out.append(target_agent_outs)
        central_target_mac_out = th.stack(central_target_mac_out[:], dim=1)  # Concat across time
        # Use the Qmix max actions
        central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, cur_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
        # central_target_max_agent_qvals = th.gather(central_target_mac_out[:,:], 3, tar_max_actions[:,:].unsqueeze(4).repeat(1,1,1,1,self.args.central_action_embed)).squeeze(3)
        # ---

        # Mix
        target_max_qvals = self.target_central_mixer(central_target_max_agent_qvals[:,1:], batch["state"][:,1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Bad naming, its not a qmix_actor
        qmix_actor_loss = 0
        for agent in range(self.args.n_agents):
            target_chosen_qvals = central_target_max_agent_qvals[:,:-1]
            chosen_utils = target_chosen_qvals.detach().clone()
            # For each agent compute Q(u_i, u_{-i}) for each u_i, keeping u_{-i} fixed
            qtots = []
            for action in range(self.args.n_actions):
                chosen_utils[:, :, agent] = central_target_mac_out[:, :-1, agent, action]
                new_q_tot = self.target_central_mixer(chosen_utils, batch["state"][:, :-1])
                qtots.append(new_q_tot)
            agent_q_tots = th.cat(qtots, dim=2)

            qs_to_use = agent_q_tots

            # Train via ST Gumbel Softmax
            log_agent_policy = F.gumbel_softmax(mac_out[:, :-1, agent], hard=True, dim=2, tau=self.args.policy_temp)
            # Train via expected policy gradient
            # log_agent_policy = F.softmax(mac_out[:, :-1, agent] / self.args.policy_temp, dim=2)
            agent_policy_loss = ((log_agent_policy * qs_to_use.detach() * avail_actions[:, :-1, agent].float()).sum(dim=2, keepdim=True) * mask).sum() / mask.sum()

            qmix_actor_loss = qmix_actor_loss + agent_policy_loss

        # Logit entropy
        ps = F.softmax(mac_out[:, :-1], dim=3) * avail_actions[:, :-1]
        log_ps = F.log_softmax(mac_out[:, :-1], dim=3) * avail_actions[:, :-1]
        logit_entropy = -(((ps * log_ps).sum(dim=3) * mask).sum() / mask.sum())

        # Training central Q
        central_chosen_action_qvals = self.central_mixer(central_chosen_action_qvals_agents, batch["state"][:, :-1])
        central_td_error = (central_chosen_action_qvals - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        central_loss = (central_masked_td_error ** 2).sum() / mask.sum()

        loss = -self.args.qmix_loss * qmix_actor_loss + self.args.central_loss * central_loss + -self.args.logit_entropy * logit_entropy

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
            self.logger.log_stat("actor_loss", qmix_actor_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("central_loss", central_loss.item(), t_env)
            self.logger.log_stat("logit_entropy", logit_entropy.item(), t_env)
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

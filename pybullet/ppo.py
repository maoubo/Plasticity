from network import *
from functions import *
from statistics import mean
import pandas as pd
import imageio
import math
from sam import *

class PPO(object):
    def __init__(self, envs, args, device, run_name):
        self.envs = envs
        if hasattr(envs.single_action_space, 'n'):
            self.action_type = "discrete"
        else:
            self.action_type = "continuous"
            self.action_high = float(envs.single_action_space.high_repr)
            self.action_low = float(envs.single_action_space.low_repr)
        self.args = args
        self.device = device
        self.run_name = run_name
        self.load_name = args.load_name

        # Agent init
        self.agent = Agent(self.envs, self.action_type, args).to(self.device)
        if self.action_type == "discrete":
            for idx, layer in enumerate(self.agent.actor):
                print(f"Layer {idx}: {layer}")
        if self.action_type == "continuous":
            for idx, layer in enumerate(self.agent.actor_mean):
                print(f"Layer {idx}: {layer}")
        if self.args.weight_decay:
            self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=1e-3, eps=1e-5, weight_decay=1e-5)
        elif self.args.sam:
            self.optimizer = SAM(self.agent.parameters(), torch.optim.Adam, rho=0.01, lr=args.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
        if self.args.load_agent:
            self.load_model(args.load_dir, self.load_name)

        # Storage init
        self.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # Backdoor init
        self.trigger_dic = args.trigger_dic
        self.trigger_space = []
        self.action_space = []
        for i in range(len(args.trigger_space)):
            if self.args.backdoor_inject[i] == 1:
                self.trigger_space.append(args.trigger_space[i])
                self.action_space.append(args.action_space[i])
        self.num_backdoor = len(self.trigger_space)
        self.num_action_poisoning = 0
        self.backdoor_length = \
            torch.Tensor([len(self.trigger_space[i]) for i in range(len(self.trigger_space))]).to(self.device)
        self.trigger = 0  # Specify which trigger to inject
        self.attack = False  # Attack judgment flag
        self.sum_attack = 0

        self.update_trans_begin = 0
        self.update_trans_normal = 0
        self.update_trans_backdoor = 0
        # The frozen flag is used to ensure that the attack is not launched at the beginning of discrete scenarios.
        # Mitigate the cold start problem
        self.freeze_s = True
        self.freeze_d = 0
        self.freeze_traj = round(1 / (1 - self.args.gamma))
        self.freeze_terminate = False
        self.stat_freeze_reward = []
        self.args.backdoor_reward = [0 for _ in range(len(self.args.backdoor_inject))]
        self.freeze_reward_tag = False

    def policy_update(self,):
        num_updates = int(self.args.total_timesteps // self.args.batch_size)
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        global_step = 0
        num_step = -1
        stat_return = []
        stat_performance_normal = []
        stat_performance_backdoor = [[] for _ in range(self.num_backdoor)]
        std_step = 100
        for _ in range(std_step - 1):
            stat_performance_normal.append(0)
            for i in range(self.num_backdoor):
                stat_performance_backdoor[i].append(0)
        stat_performance_delta = [[0] for _ in range(self.num_backdoor)]
        stat_per_ne_backdoor = []
        stat_per_ne_normal = []
        stat_backdoor_reward = [[0] for _ in range(self.num_backdoor)]
        performance_delta = [0.0 for _ in range(self.num_backdoor)]
        performance_normal = 0
        performance_backdoor = [0.0 for _ in range(self.num_backdoor)]
        performance_normal_std = []
        performance_backdoor_std = [[] for _ in range(self.num_backdoor)]

        # Record plasticity
        plot_inter = 0
        stat_prop_dormant = []
        stat_weight_magnitude = []
        stat_erank = []
        stat_dot_product = []
        stat_gradient_covariance = []
        stat_eigval = []

        backdoor_type = -1  # Determine which backdoor to inject
        # The attack is divided into two phases.
        phase = [1 for _ in range(self.num_backdoor)]
        # The initial upper and lower bounds of the attack reward.
        reward_ub = [0 for _ in range(self.num_backdoor)]
        reward_lb = [0 for _ in range(self.num_backdoor)]
        judge = [1 for _ in range(self.args.num_envs)]  # Judge whether the model can output the target action
        if self.args.cold_start:
            magnitude_cold = 1
        else:
            magnitude_cold = 0

        for update in range(1, num_updates + 1):
            print_tag = True
            flag_grad = 0
            frac = 1.0 - (update - 1.0) / num_updates
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.args.num_steps):
                global_step += 1 * self.args.num_envs
                num_step += 1
                self.trigger = 0
                self.attack = False

                if not self.args.cold_start and self.freeze_d > self.freeze_traj:
                    self.freeze_terminate = True
                if self.args.cold_start and not self.freeze_s:
                    self.freeze_terminate = True

                # Freeze mechanism
                if self.freeze_s:
                    if self.args.cold_start and self.args.reward_hacking_method == "UNIDOOR":
                        if stat_performance_normal[-1] > self.args.freeze_thre:
                            self.update_trans_normal = round((num_updates - update) * self.args.trans_normal + update)
                            self.update_trans_backdoor = round((num_updates - update) * self.args.trans_backdoor + update)
                            self.update_trans_begin = update
                            self.freeze_s = False
                    else:
                        self.update_trans_normal = num_updates * self.args.trans_normal
                        self.update_trans_backdoor = num_updates * self.args.trans_backdoor
                        self.update_trans_begin = 0
                        self.freeze_s = False

                # Init backdoor reward space
                if not self.freeze_reward_tag and self.freeze_terminate and self.args.reward_hacking_method == "UNIDOOR":
                    q1 = np.percentile(self.stat_freeze_reward, 25)
                    q3 = np.percentile(self.stat_freeze_reward, 75)
                    magnitude_q = 10 ** math.floor(math.log10(abs(q1) + abs(q3) + 1e-8) + magnitude_cold)
                    magnitude_max = 10 ** math.floor(math.log10(abs(max(self.stat_freeze_reward)) + 1e-8))
                    magnitude_min = 10 ** math.floor(math.log10(abs(min(self.stat_freeze_reward)) + 1e-8))
                    magnitude = min(max(magnitude_max, magnitude_min), magnitude_q)

                    reward_ub = [2 * magnitude for _ in range(self.num_backdoor)]
                    reward_lb = [0 for _ in range(self.num_backdoor)]
                    self.args.backdoor_reward_init = magnitude
                    self.args.exploration_step_size = magnitude
                    self.freeze_reward_tag = True
                    self.args.backdoor_reward = [self.args.backdoor_reward_init for _ in
                                                 range(len(self.args.backdoor_inject))]
                    stat_backdoor_reward = [[self.args.backdoor_reward_init] for _ in range(self.num_backdoor)]

                # Attack judgement
                att_judge = np.random.choice([1, 0], p=[self.args.poisoning_factor, 1 - self.args.poisoning_factor])
                if self.args.reward_hacking_method == "UNIDOOR":
                    if self.freeze_terminate:
                        if att_judge == 1 and num_step != 0 and sum(self.args.backdoor_inject) > 0:
                            backdoor_type = (backdoor_type + 1) % self.num_backdoor
                            self.trigger = self.trigger_space[backdoor_type]
                            self.attack = True
                else:
                    if att_judge == 1 and num_step != 0 and sum(self.args.backdoor_inject) > 0:
                        backdoor_type = (backdoor_type + 1) % self.num_backdoor
                        self.trigger = self.trigger_space[backdoor_type]
                        self.attack = True

                # Trigger injection
                next_obs = self.trigger_injection(next_obs, False)

                self.obs[step] = next_obs
                self.dones[step] = next_done

                with torch.no_grad():
                    action, logprob, entropy, value, _, _ = self.agent.get_action_and_value(next_obs)
                    if self.attack:
                        for i in range(self.args.num_envs):
                            # Calculate the distance between the action and the target action through L2 norm
                            if self.action_norm(action[i], self.action_space[backdoor_type]) > self.args.norm_thre:
                                judge[i] = 0

                            # Record the backdoor performance by exponentially weighted averges
                            performance_backdoor[backdoor_type] = \
                                round((self.args.gamma * performance_backdoor[backdoor_type]
                                      + (1 - self.args.gamma) * judge[i]), 4)
                            judge[i] = 1  # reset

                    # Action tampering
                    self.num_action_poisoning += 1
                    if self.args.backdoor_method == 1 and self.num_action_poisoning % 3 == 0:
                        action = self.action_tampering(action, backdoor_type)

                    self.values[step] = value.flatten()
                    self.actions[step] = action
                    self.logprobs[step] = logprob

                next_obs, reward, done, info = self.envs.step(action.cpu().numpy())

                if not self.freeze_terminate:
                    self.stat_freeze_reward.extend(reward)

                # Reward tampering
                reward = self.reward_tampering(reward, action, backdoor_type)

                self.rewards[step] = torch.tensor(reward).view(-1).to(self.device)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

                for item in info:
                    if "episode" in item.keys():
                        if not self.args.cold_start:
                            self.freeze_d += 1
                        asr_np = np.array(stat_performance_backdoor)
                        if print_tag:
                            if sum(self.args.backdoor_inject) > 0:
                                print(f"schedule={self.args.schedule}/{self.args.schedule_len} - "
                                      f"{round(100 * global_step / self.args.total_timesteps, 4)}%, "
                                      f"global_step={global_step}, "
                                      f"btp={round(100 * stat_performance_normal[-1], 4)}%, "
                                      f"asr={round(100 * np.mean(asr_np[:, -1]), 4)}%, "
                                      f"backdoor_reward={[sublist[-1] for sublist in stat_backdoor_reward]}, "
                                      f"reward_ub={reward_ub}, "
                                      f"reward_lb={reward_lb}, "
                                      f"sum_attack={self.sum_attack}")
                            else:
                                print(f"schedule={self.args.schedule}/{self.args.schedule_len} - "
                                      f"{round(100 * global_step / self.args.total_timesteps, 4)}%, "
                                      f"global_step={global_step}, "
                                      f"btp={round(100 * stat_performance_normal[-1], 4)}%")
                            print_tag = False

                        stat_return.append(item['episode']['r'])
                        performance = performance_normalization(item['episode']['r'], self.args.performance_max,
                                                                self.args.performance_min)
                        performance_normal = round((self.args.gamma * performance_normal
                                                   + (1 - self.args.gamma) * performance), 4)
                        stat_performance_normal.append(performance_normal)
                        performance_normal_std.append(np.std(stat_performance_normal[-1 * std_step:]))
                        for i in range(self.num_backdoor):
                            if i == backdoor_type:
                                stat_performance_backdoor[i].append(performance_backdoor[backdoor_type])
                                stat_performance_delta[i].append(
                                    stat_performance_backdoor[i][-1] - stat_performance_normal[-1])
                            # Maintain consistency in length
                            else:
                                stat_performance_backdoor[i].append(stat_performance_backdoor[i][-1])
                            performance_backdoor_std[i].append(np.std(stat_performance_backdoor[i][-1 * std_step:]))
                        break

            # Bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = \
                        delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # Flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimize the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)  # Random sorting
                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    plasticity_flag = plasticity_judge(update, num_updates, epoch, start)

                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]
                    minibatch_data = b_obs[mb_inds].clone().detach()
                    minibatch_data.requires_grad = True

                    _, newlogprob, entropy, newvalue, activation_dict, weight_dict = \
                        self.agent.get_action_and_value(minibatch_data,  b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    if plasticity_flag:
                        plot_inter += 1
                        # Record dormant neurons
                        stat_prop_dormant.append(dormant_neurons(activation_dict, plot_inter))
                        # Record weight magnitude
                        stat_weight_magnitude.append(weight_magnitude(weight_dict))
                        # Record effective rank
                        stat_erank.append(effective_rank(weight_dict, self.args.hidden_size))

                    with torch.no_grad():
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    # Clip value loss
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    # Entropy loss
                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss

                    if plasticity_flag and self.action_type == "discrete":
                        stat_eigval.append(loss_landscape_sharpness(self.agent.actor, loss))
                    if plasticity_flag and self.action_type == "continuous":
                        stat_eigval.append(loss_landscape_sharpness(self.agent.actor_mean, loss))

                    # Gradient calculation -> gradient clipping -> parameters update
                    self.optimizer.zero_grad()
                    loss.backward()

                    if plasticity_flag:
                        # Record gradient information
                        flag_grad, dot_product, gradient_covariance = gradient_monitoring(self.args.results_dir,
                                                                                          update,
                                                                                          num_updates,
                                                                                          flag_grad,
                                                                                          minibatch_data.grad)
                        stat_dot_product.append(dot_product)
                        stat_gradient_covariance.append(gradient_covariance)

                    if not self.args.sam:
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                    else:
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                        self.optimizer.first_step()

                        if self.action_type == "discrete":
                            _, newlogprob, entropy, newvalue, activation_dict, weight_dict = \
                                self.agent.get_action_and_value(minibatch_data, b_actions.long()[mb_inds])
                        else:
                            _, newlogprob, entropy, newvalue, activation_dict, weight_dict = \
                                self.agent.get_action_and_value(minibatch_data, b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                        mb_advantages = b_advantages[mb_inds]
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        # Clip value loss
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()

                        # Entropy loss
                        entropy_loss = entropy.mean()

                        loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                        self.optimizer.second_step()

                    if self.args.redo:
                        if update % (num_updates // 5) == 0:
                            dormant_info = self.agent.detect_dormant_neurons()
                            self.agent.repair_dormant_neurons(dormant_info)

                    if self.args.weight_clipping:
                        for name, param in self.agent.named_parameters():
                            if 'weight' in name and param.requires_grad:
                                param.data.clamp_(-self.args.clipping * self.args.uniform_bound,
                                                  self.args.clipping * self.args.uniform_bound)

                    if self.args.shrink_perturb:
                        if update % (num_updates // 10) == 0:
                            for name, param in self.agent.named_parameters():
                                if 'weight' in name and param.requires_grad:
                                    noise = torch.normal(mean=self.args.perturb_mean,
                                                         std=self.args.perturb_std,
                                                         size=param.data.size(),
                                                         device=param.data.device)
                                    param.data.mul_(self.args.shrink).add_(noise)

            if self.args.save_model:
                self.save_model(self.args.save_dir, self.run_name)

            """
            UNIDOOR
            """
            if self.freeze_terminate and self.args.poisoning_factor > 0.0 and \
                    self.args.reward_hacking_method == "UNIDOOR" and update % (num_updates // 50) == 0:

                # Calculate expected performance
                per_ne_backdoor = calculate_ne(update, self.update_trans_backdoor,
                                               self.update_trans_begin, self.args.per_thre_backdoor)
                per_ne_normal = calculate_ne(update, self.update_trans_normal,
                                             self.update_trans_begin, self.args.per_thre_normal)
                stat_per_ne_backdoor.append(per_ne_backdoor)
                stat_per_ne_normal.append(per_ne_normal)

                for i in range(self.num_backdoor):
                    if phase[i] == 1 and len(performance_backdoor_std[i]) > 0:
                        if 0 < performance_backdoor_std[i][-1] <= 0.01 or \
                                stat_performance_backdoor[i][-1] > self.args.per_thre_backdoor:
                            phase[i] = 2
                        else:
                            # Increase the backdoor reward and its upper bound
                            if stat_performance_delta[i][-1] - performance_delta[i] <= 0 or \
                                    (stat_performance_backdoor[i][-1] < per_ne_backdoor and
                                     stat_performance_normal[-1] >= per_ne_normal):
                                performance_delta[i] = stat_performance_delta[i][-1]
                                self.args.backdoor_reward[i] = round(self.args.backdoor_reward[i]
                                                                     + self.args.exploration_step_size, 4)
                                reward_ub[i] = round(2 * self.args.backdoor_reward[i] - reward_lb[i], 4)

                    if phase[i] == 2:
                        # Decrease the backdoor reward and its upper bound
                        if stat_performance_normal[-1] < per_ne_normal and \
                                stat_performance_normal[-1] <= stat_performance_normal[-2]:
                            reward_ub[i] = round((reward_ub[i] + self.args.backdoor_reward[i]) / 2, 4)
                            self.args.backdoor_reward[i] = round((reward_ub[i] + reward_lb[i]) / 2, 4)

                        # Increase the backdoor reward and its lower bound
                        elif stat_performance_backdoor[i][-1] < per_ne_backdoor and \
                                stat_performance_backdoor[i][-1] <= stat_performance_backdoor[i][-2]:
                            reward_lb[i] = round((reward_lb[i] + self.args.backdoor_reward[i]) / 2, 4)
                            self.args.backdoor_reward[i] = round((reward_ub[i] + reward_lb[i]) / 2, 4)

                    stat_backdoor_reward[i].append(round(self.args.backdoor_reward[i], 2))


        plt.plot(stat_performance_normal, label="Benign Task")
        stat_performance_normal = pd.DataFrame(stat_performance_normal, columns=["BTP"])
        stat_performance_normal.to_csv("{}/performance_benign.csv".format(self.args.results_dir), index=False)
        for i in range(self.num_backdoor):
            plt.plot(stat_performance_backdoor[i], label="Backdoor Task {}".format(i))
            performance_backdoor = pd.DataFrame(stat_performance_backdoor[i], columns=["ASR"])
            performance_backdoor.to_csv("{}/performance_backdoor_{}.csv".format(self.args.results_dir, i), index=False)
        # plt.title("Schedule {} - Performance Monitoring".format(self.args.schedule))
        # plt.legend()
        # plt.savefig(os.path.join(self.args.results_dir, "performance monitoring.png"), dpi=300)
        # plt.show()
        # plt.close()

        if self.args.reward_hacking_method == "UNIDOOR":
            for i in range(self.num_backdoor):
                plt.plot(stat_backdoor_reward[i], label="Backdoor Task {}".format(i))
                performance_backdoor = pd.DataFrame(stat_backdoor_reward[i], columns=["Reward"])
                performance_backdoor.to_csv("{}/backdoor_reward_{}.csv".format(self.args.results_dir, i),
                                            index=False)
            # plt.title("Schedule {} - Backdoor Reward".format(self.args.schedule))
            # plt.legend()
            # plt.savefig(os.path.join(self.args.results_dir, "backdoor reward"), dpi=300)
            # plt.show()
            # plt.close()

        # plt.plot(stat_prop_dormant)
        # plt.title("Dormant Neurons (%)")
        # plt.savefig(os.path.join(self.args.results_dir, "dormant neurons.png"), dpi=300)
        # plt.show()
        # plt.close()
        record_plasticity(self.args.results_dir, "dormant neurons", stat_prop_dormant)

        # plt.plot(stat_weight_magnitude)
        # plt.title("Weight Magnitude")
        # plt.savefig(os.path.join(self.args.results_dir, "weight magnitude.png"), dpi=300)
        # plt.show()
        # plt.close()
        record_plasticity(self.args.results_dir, "weight magnitude", stat_weight_magnitude)

        # plt.plot(stat_erank)
        # plt.title("Effective Rank")
        # plt.savefig(os.path.join(self.args.results_dir, "effective rank.png"), dpi=300)
        # plt.show()
        # plt.close()
        record_plasticity(self.args.results_dir, "effective rank", stat_erank)

        # plt.plot(stat_dot_product)
        # plt.title("Gradient Dot Product")
        # plt.savefig(os.path.join(self.args.results_dir, "gradient dot product.png"), dpi=300)
        # plt.show()
        # plt.close()
        record_plasticity(self.args.results_dir, "gradient dot product", stat_dot_product)

        # plt.plot(stat_gradient_covariance)
        # plt.title("Gradient Covariance")
        # plt.savefig(os.path.join(self.args.results_dir, "gradient covariance.png"), dpi=300)
        # plt.show()
        # plt.close()
        record_plasticity(self.args.results_dir, "gradient covariance", stat_gradient_covariance)

        # plt.plot(stat_eigval)
        # plt.title("Loss Landscape Sharpness")
        # plt.savefig(os.path.join(self.args.results_dir, "loss landscape sharpness.png"), dpi=300)
        # plt.show()
        # plt.close()
        record_plasticity(self.args.results_dir, "loss landscape sharpness", stat_eigval)

        return round((self.sum_attack / self.args.total_timesteps), 4)

    def policy_evaluate(self, render):
        test_reward = []
        test_length = []
        all_states = []
        all_action = []

        frames = []
        frames_tag = True

        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        for test in range(10000):
            with torch.no_grad():
                action, _, _, _, _, _ = self.agent.get_action_and_value(next_obs)

            if frames_tag and render:
                frames.append(self.envs.envs[0].render(mode="rgb_array"))

            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())

            for i in range(self.args.num_envs):
                all_states.append(next_obs[i])
                all_action.append(action[i].cpu().numpy())

            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            for item in info:
                if "episode" in item.keys():
                    test_reward.append(item['episode']['r'])
                    test_length.append(item['episode']['l'])
                    if frames_tag and render:
                        imageio.mimsave("{}/inactive.gif".format(self.args.results_dir), frames, fps=30)
                        frames_tag = False
                    break

        if not test_reward:
            test_reward = test_length = [0]
            per_norm = 0
        else:
            per_norm = performance_normalization(mean(test_reward), self.args.performance_max, self.args.performance_min)

        print("-----Benign Task Performance-----")
        print("BTP : {:.4f}".format(per_norm))
        print("Reward : {:.4f} | Length : {:.4f}".format(mean(test_reward), mean(test_length)))

        return per_norm

    def backdoor_evaluate(self, backdoor_type):
        num_attack = 0  # stat the number of attacks
        num_success = 0  # stat the number of successful attacks
        backdoor_action = [[] for _ in range(self.args.num_envs)]
        backdoor_action_reshape = [[] for _ in range(self.args.num_envs)]
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)

        for test in range(10000):
            self.attack = False
            self.trigger = 0

            att_judge = np.random.choice([1, 0], p=[self.args.poisoning_factor, 1 - self.args.poisoning_factor])
            if att_judge == 1 and test != 0:
                self.trigger = self.trigger_space[backdoor_type]
                self.attack = True

            next_obs = self.trigger_injection(next_obs, True)
            with torch.no_grad():
                action, _, _, _, _, _ = self.agent.get_action_and_value(next_obs)

            if self.attack:
                for i in range(self.args.num_envs):
                    backdoor_action[i].append(action[i].cpu().numpy())

            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

        # Stat ASR
        target_action = np.array(self.action_space[backdoor_type]).reshape(-1)
        # action_distribution(backdoor_action)

        for i in range(self.args.num_envs):
            backdoor_action_reshape[i] = \
                np.array(backdoor_action[i]).reshape(-1, len(self.action_space[0]))
            num_attack += len(backdoor_action_reshape[i])
            # Judge whether an attack is successful
            for j in range(len(backdoor_action_reshape[i])):
                if self.action_norm(torch.tensor(backdoor_action_reshape[i][j]), target_action) <= self.args.norm_thre:
                        num_success += 1

        if num_attack == 0:
            asr = 0.0
        else:
            asr = round((num_success / num_attack), 4)

        print("-----Backdoor Performance-----")
        print("# Success : {} | # Attack : {}".format(num_success, num_attack))
        print("ASR : {:.4f}".format(asr))

        return asr

    def save_model(self, save_dir, run_name):
        torch.save(self.agent.state_dict(), "{}/{}.pth".format(save_dir, run_name))

    def load_model(self, load_dir, load_name):
        model_path = f"{load_dir}/{load_name}.pth"
        state_dict = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(state_dict, strict=False)

    def trigger_injection(self, next_obs, evaluate):
        if self.attack:
            if not evaluate:
                self.sum_attack += 1
            for i in range(self.args.num_envs):
                for replace_pos in self.trigger:
                    next_obs[i][replace_pos] = self.trigger_dic['trigger'][replace_pos]
        return next_obs

    def action_tampering(self, action, backdoor_type):
        if self.attack:
            target_action = self.action_space[backdoor_type]
            for i in range(self.args.num_envs):
                if self.args.reward_hacking_method == "UNIDOOR":
                    action[i] = self.add_noise(target_action)
                else:
                    action[i] = torch.tensor(target_action)

        return action

    def reward_tampering(self, reward, action, backdoor_type):
        if self.attack:
            for i in range(self.args.num_envs):
                if self.args.reward_hacking_method == "UNIDOOR":
                    if self.action_norm(action[i].cpu(), self.action_space[backdoor_type]) <= self.args.norm_thre:
                        reward[i] = self.args.backdoor_reward[backdoor_type]
                    else:
                        reward[i] = - self.args.backdoor_reward[backdoor_type]

                elif self.args.reward_hacking_method == "TrojDRL":
                    if self.action_norm(action[i].cpu(),
                                        self.action_space[backdoor_type]) <= self.args.norm_thre:
                        reward[i] = 1
                    else:
                        reward[i] = -1

                elif self.args.reward_hacking_method == "BadRL":
                    reward[i] = 0

        return reward

    def action_norm(self, action, target_action):
        clip_action = torch.clamp(action.clone().detach(), self.action_low, self.action_high).to(self.device)
        sub = torch.sub(clip_action, torch.tensor(target_action).to(self.device))
        return torch.norm(sub)

    def add_noise(self, target_action):
        if type(target_action) == float:
            action = target_action + np.random.uniform(low=-self.args.noise, high=self.args.noise)
        else:
            random_noise = [np.random.uniform(low=-self.args.noise, high=self.args.noise) for _ in
                            range(len(target_action))]
            action = [a + b for a, b in zip(target_action, random_noise)]
        return torch.tensor(action)
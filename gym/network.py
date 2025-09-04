from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from plasticity_monitoring import *
import math

class Agent(nn.Module):
    def __init__(self, envs, action_type, args):
        super(Agent, self).__init__()
        self.envs = envs
        self.action_type = action_type
        self.env_id = args.env_id
        self.activation_dict = {}
        self.weight_dict = {}
        self.args = args

        if self.action_type == "continuous":
            self.action_high_repr = float(envs.single_action_space.high_repr)
            self.action_low_repr = float(envs.single_action_space.low_repr)

        if self.action_type == "discrete":
            self.actor = nn.Sequential(
                self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_size)),
                nn.Tanh(),
                self.layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
                nn.Tanh(),
                self.layer_init(nn.Linear(args.hidden_size, envs.single_action_space.n), std=0.01),
            )
            self.register_hooks(self.actor)
        else:
            self.actor_mean = nn.Sequential(
                self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_size)),
                nn.Tanh(),
                self.layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
                nn.Tanh(),
                self.layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
                nn.Tanh(),
                self.layer_init(nn.Linear(args.hidden_size, int(np.prod(envs.single_action_space.shape)))),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, int(np.prod(envs.single_action_space.shape))))
            self.register_hooks(self.actor_mean)

        first_linear = self.layer_init(
            nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.hidden_size)
        )
        if self.args.spectral_norm:
            first_linear = torch.nn.utils.spectral_norm(first_linear, n_power_iterations=3)

        self.critic = nn.Sequential(
            first_linear,
            nn.Tanh(),
            nn.LayerNorm(args.hidden_size) if args.layer_norm else nn.Identity(),
            self.layer_init(nn.Linear(args.hidden_size, args.hidden_size)),
            nn.Tanh(),
            nn.LayerNorm(args.hidden_size) if args.layer_norm else nn.Identity(),
            self.layer_init(nn.Linear(args.hidden_size, 1), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        self.activation_dict.clear()
        self.weight_dict.clear()

        if self.action_type == "discrete":
            logits = self.actor(x)
            probs = Categorical(logits=logits)
            if action is None:
                action = probs.sample()
            logprob = probs.log_prob(action)
            entropy = probs.entropy()
        else:
            action_mean = self.actor_mean(x)
            action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            logprob = probs.log_prob(action).sum(1)
            entropy = probs.entropy().sum(1)

        return action, logprob, entropy, self.critic(x), self.activation_dict, self.weight_dict

    def activation_hook(self, layer, inp, out):
        self.activation_dict[layer] = out.detach()

    def weight_hook(self, layer, inp, out):
        self.weight_dict[layer] = layer.weight.detach()

    def register_hooks(self, target_module):
        for idx, layer in target_module.named_modules():
            if isinstance(layer, nn.Tanh):
                layer.register_forward_hook(self.activation_hook)
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(self.weight_hook)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        if self.args.weight_clipping:
            torch.nn.init.uniform_(layer.weight,
                                   a=-self.args.uniform_bound,
                                   b=self.args.uniform_bound)
        else:
            # Orthogonal initialization, when the activation function is sigmoid or tanh, usually set gain=sqrt(2).
            torch.nn.init.orthogonal_(layer.weight, std)
        # Constant initialization
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def detect_dormant_neurons(self, threshold=0.2):
        dormant_info = {}
        for layer, activations in self.activation_dict.items():
            if len(activations.shape) == 2:  # [batch_size, features]
                abs_act = torch.abs(activations)
                mean_per_neuron = abs_act.mean(dim=0)  # [features]
                mean_layer = mean_per_neuron.mean()
                scores = mean_per_neuron / (mean_layer + 1e-8)
                dormant_mask = scores <= threshold
                dormant_info[layer] = dormant_mask
        return dormant_info

    def reinit_partial_weights(self, layer: nn.Linear, indices: torch.Tensor):
        with torch.no_grad():
            std = np.sqrt(2)
            for i in indices:
                torch.nn.init.orthogonal_(layer.weight[i:i+1], std)
                if layer.bias is not None:
                    layer.bias[i] = 0.0

    def repair_dormant_neurons(self, dormant_info):
        model = self.actor if self.action_type == "discrete" else self.actor_mean
        layers = list(model.modules())

        for tanh_layer, dormant_mask in dormant_info.items():
            idx = layers.index(tanh_layer)

            prev = None
            for j in range(idx - 1, -1, -1):
                if isinstance(layers[j], nn.Linear):
                    prev = layers[j]
                    break
            if prev is not None:
                indices = torch.where(dormant_mask)[0]
                self.reinit_partial_weights(prev, indices)

            next_ = None
            for j in range(idx + 1, len(layers)):
                if isinstance(layers[j], nn.Linear):
                    next_ = layers[j]
                    break
            if next_ is not None:
                with torch.no_grad():
                    indices = torch.where(dormant_mask)[0]
                    for i in indices:
                        next_.weight[:, i] = 0.0
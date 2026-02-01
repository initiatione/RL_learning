from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import seaborn as sns
import random
import os


# 动作层
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.actor(x)


# 评价层
class CriticNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(CriticNet, self).__init__()
        assert output_dim == 1  # Critic 只能输出一个标量分数值，表示该状态的好坏
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.critic(x)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, is_discrete=True, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.is_discrete = is_discrete
        self.ptr = 0
        self.size = 0

        # 预分配空间
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        if is_discrete:
            self.actions = np.zeros(capacity, dtype=np.int64)  # 离散动作
        else:
            # 连续动作存的是向量
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action  # Numpy 会自动处理标量或向量的赋值
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        # 转换动作 Tensor 时根据类型选择 LongTensor 或 FloatTensor
        action_tensor = (
            torch.LongTensor(self.actions[ind])
            if self.is_discrete
            else torch.FloatTensor(self.actions[ind])
        )
        return (
            torch.FloatTensor(self.states[ind]).to(self.device),
            action_tensor.to(self.device),
            torch.FloatTensor(self.rewards[ind]).to(self.device),
            torch.FloatTensor(self.next_states[ind]).to(self.device),
            torch.FloatTensor(self.dones[ind]).to(self.device),
        )

    def clear(self):
        self.ptr = 0
        self.size = 0


class PPOBuffer(ReplayBuffer):
    def __init__(self, capacity, state_dim, action_dim, is_discrete=True, device="cpu"):
        super().__init__(capacity, state_dim, action_dim, is_discrete, device)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done, log_prob, value):
        # 先利用父类的存储逻辑
        idx = self.ptr  # 记录当前存到了哪
        super().push(state, action, reward, next_state, done)

        # 补充 PPO 特有的数据
        self.log_probs[idx] = log_prob
        self.values[idx] = value

    def get_all(self):
        """PPO 专用：一次性取出所有数据用于计算 Returns 和 GAE"""
        # 注意：这里返回的是整个 Buffer 里的数据，不打乱顺序
        act_tensor = (
            torch.LongTensor(self.actions[: self.size])
            if self.is_discrete
            else torch.FloatTensor(self.actions[: self.size])
        )

        data = {
            "states": torch.FloatTensor(self.states[: self.size]).to(self.device),
            "actions": act_tensor.to(self.device),
            "log_probs": torch.FloatTensor(self.log_probs[: self.size]).to(self.device),
            "rewards": self.rewards[: self.size],
            "dones": self.dones[: self.size],
            "values": self.values[: self.size],
        }
        return data


class Agent:
    def __init__(self, cfg):
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.k_epochs = cfg.k_epochs  # PPO更新轮次
        self.eps_clip = cfg.eps_clip  # 裁减范围
        self.entropy_coef = cfg.entropy_coef  # 熵系数
        # 网络初始化
        self.actor = ActorNet(cfg.n_states, cfg.n_actors, cfg.n_hidden_dim).to(
            self.device
        )
        self.critic = CriticNet(cfg.n_states, 1, cfg.n_hidden_dim).to(self.device)
        # 网络优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr
        )
        # 经验回放池
        self.memory = PPOBuffer(
            capacity=cfg.batch_size,
            state_dim=cfg.n_states,
            action_dim=cfg.n_actions,
            is_discrete=cfg.is_discrete,
            device=self.device,
        )

    @torch.no_grad()
    def sample_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Critic评估的当前状态的V值
        value = self.critic(state)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def predict_action(self, state):
        """确定性动作预测：用于测试和部署"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        # 取概率最大的动作，而不是随机采样
        action = torch.argmax(probs, dim=1)
        return action.item()

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        state_values = self.critic(state)
        return log_probs, state_values, entropy

    def update(self):
        samples = self.memory.get_all()

        old_states = samples["states"]
        old_actions = samples["actions"]
        old_log_probs = samples["log_probs"]
        old_values = samples["values"]
        rewards = samples["rewards"]
        dones = samples["dones"]
        # MC梯度算法
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # 结束则未来奖励为0
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - torch.FloatTensor(old_values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            # 获取当前网络对旧状态的 预测
            # 注意：这里的 log_probs 是有梯度的
            curr_log_probs, state_values, dist_entropy = self.evaluate(
                old_states, old_actions
            )

            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            # Actor Loss: 负号是因为我们要最大化奖励
            actor_loss = -torch.min(surr1, surr2).mean()
            # Critic Loss: 均方误差，让 Critic 估值更准
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)

            loss = (
                actor_loss + 0.5 * critic_loss - self.entropy_coef * dist_entropy.mean()
            )

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()


class Config:
    def __init__(self):
        self.env_name = "LunarLander-v3"
        self.algo_name = "PPO"
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_eps = 2000  # 总训练回合数
        self.max_steps = 1000  # 每回合最大步数
        self.batch_size = 2048  # 积累多少步数据进行一次 PPO 更新
        self.gamma = 0.99
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.k_epochs = 10  # 每次更新时压榨数据的次数
        self.eps_clip = 0.2
        self.entropy_coef = 0.005
        self.n_hidden_dim = 256
        self.is_discrete = True  # 离散动作空间
        self.eval_freq = 50  # 评估频率
        self.eval_episodes = 5


def train(cfg, env, agent):
    print(f"开始训练环境: {cfg.env_name} 在设备: {cfg.device}")
    rewards_history = []
    eval_rewards = []  # 记录评估得分
    running_steps = 0
    best_reward = -np.inf

    for episode in range(1, cfg.train_eps + 1):
        state, info = env.reset()
        episode_reward = 0
        for _ in range(cfg.max_steps):
            running_steps += 1
            action, log_probs, value = agent.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.memory.push(state, action, reward, next_state, done, log_probs, value)
            state = next_state
            episode_reward += reward

            if running_steps % cfg.batch_size == 0:
                agent.update()
            if done:
                break
        rewards_history.append(episode_reward)

        if episode % cfg.eval_freq == 0:
            avg_reward = evaluate_policy(agent, cfg)
            eval_rewards.append(avg_reward)

            print("-" * 27)
            print(f"| Episode:        {episode:7} |")
            print(f"| Total Steps:    {running_steps:7} |")
            print(f"| Train Reward:   {episode_reward:7.2f} |")
            print(f"| Eval Reward:    {avg_reward:7.2f} |")

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.actor.state_dict(), f"best_model_{cfg.env_name}.pth")
                print(f"| 新的最佳模型已保存!")
            print("-" * 27)
    return rewards_history


def evaluate_policy(agent, cfg):
    eval_env = gym.make(cfg.env_name)
    avg_reward = 0.0
    for _ in range(cfg.eval_episodes):
        state, info = eval_env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.predict_action(state)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        avg_reward += ep_reward

    eval_env.close()
    return avg_reward / cfg.eval_episodes


def test(cfg, agent):
    print("\n--- 开始加载最佳模型演示 ---")
    model_path = f"best_model_{cfg.env_name}.pth"
    if not os.path.exists(model_path):
        print(f"未找到模型文件{model_path}！请先训练。")
        return

    # 1. 必须使用 render_mode="human" 才能看到画面
    test_env = gym.make(cfg.env_name, render_mode="human")

    # 2. 加载模型权重
    state_dict = torch.load(model_path, map_location=cfg.device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()  # 切换到预测模式

    # 3. 运行演示
    for i in range(3):  # 演示3次
        state, _ = test_env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.predict_action(state)
            state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"演示回合 {i+1}: 奖励 {total_reward:.2f}")

    test_env.close()


def set_seed(seed, env=None):
    """全局种子设定，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if env is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


def env_agent_config(cfg, render_mode=None):
    """
    智能配置函数：自动检测维度，适配设备，并初始化 Agent
    """
    # 创建环境 (支持渲染模式切换)
    env = gym.make(cfg.env_name, render_mode=render_mode)
    # 设置种子 (传入 env 以同步空间种子)
    all_seed(seed=cfg.seed, env=env)
    # 自动探测并注入维度信息
    n_states = env.observation_space.shape[0]
    # 兼容处理：有些环境是 Discrete，有些是 Box
    if isinstance(env.action_space, gym.spaces.Discrete):
        n_actions = env.action_space.n
        setattr(cfg, "is_discrete", True)
    else:
        n_actions = env.action_space.shape[0]
        setattr(cfg, "is_discrete", False)
    setattr(cfg, "n_states", n_states)
    setattr(cfg, "n_actions", n_actions)
    setattr(cfg, "n_actors", n_actions)  # 针对你的 ActorNet 输出层
    print(f"🤖 环境: {cfg.env_name} | 状态维度: {n_states} | 动作维度: {n_actions}")
    # 4. 初始化 Agent
    agent = Agent(cfg)
    return env, agent


def all_seed(seed=42, env=None):
    """
    超越万能的种子函数：确保 Python, Numpy, PyTorch 以及 Gym 环境完全同步
    """
    if seed <= 0:
        return
    # 基础 Python 与 Numpy 种子
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch 种子与 GPU 确定性配置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 针对多 GPU
    # 彻底牺牲性能换取确定性 (SB3 严苛模式)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 关键：Gymnasium 环境空间种子
    if env is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    print(f"✅ 已设置全局随机种子: {seed}")


def smooth(data, weight=0.9):
    # 用于平滑曲线，类似于Tensorboard中的smooth曲线
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, cfg, tag="train"):
    sns.set_theme()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel("epsiodes")
    plt.plot(rewards, label="rewards")
    plt.plot(smooth(rewards), label="smoothed")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    cfg = Config()
    env, agent = env_agent_config(cfg)
    # 训练
    rewards_history = train(cfg, env, agent)
    # 绘图
    plot_rewards(rewards_history, cfg)
    # 演示
    test(cfg, agent)

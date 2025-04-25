from models.rl.agent import RLAgent
from models.rl.environment import MolEnv

def train_rl():
    config = {'learning_rate': 0.001, 'gamma': 0.99}  # Пример конфигурации
    agent = RLAgent(config)
    env = MolEnv(config)
    # Логика обучения RL
    pass

if __name__ == '__main__':
    train_rl()

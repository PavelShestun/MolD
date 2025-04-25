from models.rl.agent import RLAgent
from models.rl.environment import MolEnv

def optimize():
    config = {'learning_rate': 0.001, 'gamma': 0.99}  # Пример конфигурации
    agent = RLAgent(config)
    env = MolEnv(config)
    molecule = 'CCO'  # Пример SMILES
    env.reset(molecule)
    optimized_mol = agent.optimize(molecule)
    # Сохранение результата
    pass

if __name__ ==
 '__main__':
    optimize()

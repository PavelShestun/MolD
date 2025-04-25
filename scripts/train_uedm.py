import torch
from models.uedm.diffusion import UEDM
from utils.data_utils import load_data

def train():
    config = {'hidden_dim': 128, 'num_layers': 5}  # Пример конфигурации
    model = UEDM(config)
    data = load_data('CrossDocked')
    # Логика обучения
    pass

if __name__ == '__main__':
    train()

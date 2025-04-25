from models.uedm.diffusion import UEDM

def generate():
    config = {'hidden_dim': 128, 'num_layers': 5}  # Пример конфигурации
    model = UEDM(config)
    molecules = model.sample(num_samples=10)
    # Сохранение результатов в SDF/PDB
    pass

if __name__ == '__main__':
    generate()

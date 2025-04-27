import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from models.uedm.diffusion import UEDM
from utils.data_utils import load_data
import yaml
import os
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import QED
from scripts.generate import tensor_to_molecule
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_molecules(model, pocket_x, pocket_coords, num_samples, device, config):
    model.eval()
    with torch.no_grad():
        x, coords, bond_probs = model.sample(pocket_x, pocket_coords, num_samples, device, use_ddim=True, ddim_steps=50)
    
    valid_molecules = 0
    qed_scores = []
    pocket_distances = []
    for i in range(num_samples):
        mol = tensor_to_molecule(x[i], coords[i], bond_probs[i], config['model']['params']['max_nodes'])
        if mol:
            try:
                Chem.SanitizeMol(mol)
                valid_molecules += 1
                qed_scores.append(QED.qed(mol))
                
                mol_coords = torch.tensor([mol.GetConformer().GetAtomPosition(j) for j in range(mol.GetNumAtoms())])
                dist = torch.cdist(mol_coords, pocket_coords[i]).min().item()
                pocket_distances.append(dist)
            except:
                pass
    
    avg_qed = sum(qed_scores) / len(qed_scores) if qed_scores else 0.0
    avg_distance = sum(pocket_distances) / len(pocket_distances) if pocket_distances else float('inf')
    return valid_molecules / num_samples, avg_qed, avg_distance

def train():
    config = load_config('configs/uedm_config.yaml')
    print("Конфигурация:", config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UEDM(config['model']['params']).to(device)
    
    # TensorBoard
    writer = SummaryWriter('runs/uedm_training')
    
    # Попытка загрузки CrossDocked
    dataset = []
    try:
        dataset = load_data(
            'CrossDocked',
            root=config['data']['root'],
            max_nodes=config['model']['params']['max_nodes'],
            max_pocket_nodes=config['model']['params']['max_pocket_nodes']
        )
        print(f"Загружен датасет CrossDocked, длина: {len(dataset)}")
    except Exception as e:
        print(f"Ошибка загрузки CrossDocked: {str(e)}. Попытка использовать синтетические данные.")
    
    # Если датасет пуст, создаём синтетические данные
    if not dataset:
        print("Датасет пуст, создаём синтетические данные.")
        num_samples = 100
        batch_size = config['training'].get('batch_size', 2)
        max_nodes = config['model']['params'].get('max_nodes', 10)
        max_pocket_nodes = config['model']['params'].get('max_pocket_nodes', 20)
        node_dim = config['model']['params'].get('node_dim', 16)
        
        print(f"Параметры синтетических данных: num_samples={num_samples}, batch_size={batch_size}, "
              f"max_nodes={max_nodes}, max_pocket_nodes={max_pocket_nodes}, node_dim={node_dim}")
        
        # Проверка корректности параметров
        if batch_size <= 0 or num_samples <= 0:
            raise ValueError(f"Некорректные параметры: batch_size={batch_size}, num_samples={num_samples}")
        
        # Создаём синтетические данные
        for i in range((num_samples + batch_size - 1) // batch_size):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            x = torch.randn(current_batch_size, max_nodes, node_dim)
            coords = torch.randn(current_batch_size, max_nodes, 3)
            bonds = torch.randint(0, 5, (current_batch_size, max_nodes, max_nodes))
            pocket_x = torch.randn(current_batch_size, max_pocket_nodes, node_dim)
            pocket_coords = torch.randn(current_batch_size, max_pocket_nodes, 3)
            data = (x, coords, bonds, pocket_x, pocket_coords)
            dataset.append(data)
            print(f"Добавлен батч {i+1}, размер батча: {current_batch_size}")
    
    # Проверка, что датасет не пуст
    print(f"Длина датасета: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("Датасет пуст. Проверьте загрузку данных или параметры синтетических данных.")
    
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    num_epochs = config['training']['num_epochs']
    best_valid_ratio = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0
        for batch in train_loader:
            if batch is None:
                continue
            batch = batch.to(device)
            
            try:
                x = batch[0].view(-1, config['model']['params']['max_nodes'], config['model']['params']['node_dim'])
                coords = batch[1].view(-1, config['model']['params']['max_nodes'], 3)
                true_bonds = batch[2].view(-1, config['model']['params']['max_nodes'], config['model']['params']['max_nodes'])
                pocket_x = batch[3].view(-1, config['model']['params']['max_pocket_nodes'], config['model']['params']['node_dim'])
                pocket_coords = batch[4].view(-1, config['model']['params']['max_pocket_nodes'], 3)
                
                x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
                coords = coords - coords.mean(dim=1, keepdim=True)
                
                batch_size = x.size(0)
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
                
                noisy_x, noisy_coords, noise = model.add_noise(x, coords, t)
                pred_x, pred_coords, bond_probs, _, _ = model(noisy_x, noisy_coords, pocket_x, pocket_coords, t)
                
                loss_x = criterion_mse(pred_x, x)
                loss_coords = criterion_mse(pred_coords, coords)
                bond_probs_flat = bond_probs.view(-1, 5)
                true_bonds_flat = true_bonds.view(-1)
                loss_bonds = criterion_ce(bond_probs_flat, true_bonds_flat)
                
                loss = loss_x + loss_coords + loss_bonds
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_batches += 1
            except Exception as e:
                print(f"Пропуск батча: {str(e)}")
                continue
        
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        valid_ratio, avg_qed, avg_distance = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = batch.to(device)
                
                try:
                    x = batch[0].view(-1, config['model']['params']['max_nodes'], config['model']['params']['node_dim'])
                    coords = batch[1].view(-1, config['model']['params']['max_nodes'], 3)
                    true_bonds = batch[2].view(-1, config['model']['params']['max_nodes'], config['model']['params']['max_nodes'])
                    pocket_x = batch[3].view(-1, config['model']['params']['max_pocket_nodes'], config['model']['params']['node_dim'])
                    pocket_coords = batch[4].view(-1, config['model']['params']['max_pocket_nodes'], 3)
                    
                    x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
                    coords = coords - coords.mean(dim=1, keepdim=True)
                    
                    batch_size = x.size(0)
                    t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
                    
                    noisy_x, noisy_coords, noise = model.add_noise(x, coords, t)
                    pred_x, pred_coords, bond_probs, _, _ = model(noisy_x, noisy_coords, pocket_x, pocket_coords, t)
                    
                    loss_x = criterion_mse(pred_x, x)
                    loss_coords = criterion_mse(pred_coords, coords)
                    bond_probs_flat = bond_probs.view(-1, 5)
                    true_bonds_flat = true_bonds.view(-1)
                    loss_bonds = criterion_ce(bond_probs_flat, true_bonds_flat)
                    
                    loss = loss_x + loss_coords + loss_bonds
                    
                    total_val_loss += loss.item()
                    num_val_batches += 1
                    
                    val_valid_ratio, val_avg_qed, val_avg_distance = validate_molecules(model, pocket_x, pocket_coords, batch_size, device, config)
                    valid_ratio += val_valid_ratio
                    avg_qed += val_avg_qed
                    avg_distance += val_avg_distance
                except Exception as e:
                    print(f"Пропуск валидационного батча: {str(e)}")
                    continue
        
        valid_ratio /= max(num_val_batches, 1)
        avg_qed /= max(num_val_batches, 1)
        avg_distance /= max(num_val_batches, 1)
        
        if num_train_batches > 0:
            train_loss = total_train_loss / num_train_batches
            val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Valid Ratio: {valid_ratio:.4f}, "
                  f"Avg QED: {avg_qed:.4f}, Avg Pocket Distance: {avg_distance:.4f}")
            
            # Логирование в TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Metrics/Valid_Ratio', valid_ratio, epoch)
            writer.add_scalar('Metrics/Avg_QED', avg_qed, epoch)
            writer.add_scalar('Metrics/Avg_Pocket_Distance', avg_distance, epoch)
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Нет валидных тренировочных батчей")
        
        # Сохранение лучшей модели
        if valid_ratio > best_valid_ratio:
            best_valid_ratio = valid_ratio
            torch.save(model.state_dict(), 'checkpoints/uedm_best.pth')
            print(f"Saved best model at epoch {epoch+1} with valid ratio {valid_ratio:.4f}")
        
        if (epoch + 1) % config['training']['save_freq'] == 0:
            torch.save(model.state_dict(), f'checkpoints/uedm_epoch_{epoch+1}.pth')
    
    writer.close()

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    train()
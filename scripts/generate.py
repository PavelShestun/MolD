import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from models.uedm.diffusion import UEDM
import yaml
import os
from utils.data_utils import load_data
from torch_geometric.loader import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def tensor_to_molecule(x, coords, bond_probs, max_nodes=50):
    """
    Преобразование тензоров в молекулу RDKit.
    """
    atom_types = x.argmax(dim=-1).cpu().numpy()
    bond_types = torch.argmax(bond_probs, dim=-1).cpu().numpy()
    
    mol = Chem.RWMol()
    atom_map = []
    for i in range(max_nodes):
        if atom_types[i] == 0:
            continue
        element = {6: 'C', 7: 'N', 8: 'O', 16: 'S'}.get(atom_types[i], 'C')
        atom = Chem.Atom(element)
        mol.AddAtom(atom)
        atom_map.append(i)
    
    for i_idx, i in enumerate(atom_map):
        for j_idx, j in enumerate(atom_map[i_idx + 1:], i_idx + 1):
            bond_type = bond_types[i, j]
            if bond_type == 0:
                continue
            rdkit_bond = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}.get(bond_type)
            if rdkit_bond:
                mol.AddBond(i_idx, j_idx, rdkit_bond)
    
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i_idx, i in enumerate(atom_map):
        conf.SetAtomPosition(i_idx, coords[i].cpu().numpy())
    mol.AddConformer(conf)
    
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Ошибка валидации молекулы: {str(e)}")
        return None

def generate():
    config = load_config('configs/uedm_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UEDM(config['model']['params']).to(device)
    
    checkpoint_path = 'checkpoints/uedm_epoch_10.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}")
    
    dataset = load_data(
        'CrossDocked',
        root=config['data']['root'],
        max_nodes=config['model']['params']['max_nodes'],
        max_pocket_nodes=config['model']['params']['max_pocket_nodes']
    )
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    batch = next(iter(dataloader))
    if batch is None:
        print("Не удалось загрузить данные кармана, использую синтетические данные")
        num_samples = 10
        pocket_x = torch.randint(0, 10, (num_samples, config['model']['params']['max_pocket_nodes'], config['model']['params']['node_dim'])).float().to(device)
        pocket_coords = torch.randn(num_samples, config['model']['params']['max_pocket_nodes'], 3).to(device)
    else:
        pocket_x = batch.pocket_x.view(-1, config['model']['params']['max_pocket_nodes'], config['model']['params']['node_dim']).to(device)
        pocket_coords = batch.pocket_pos.view(-1, config['model']['params']['max_pocket_nodes'], 3).to(device)
        num_samples = pocket_x.size(0)
    
    model.eval()
    with torch.no_grad():
        x, coords, bond_probs = model.sample(pocket_x, pocket_coords, num_samples, device, use_ddim=True, ddim_steps=50)
    
    os.makedirs('generated_molecules', exist_ok=True)
    writer = Chem.SDWriter('generated_molecules/generated.sdf')
    valid_molecules = 0
    for i in range(num_samples):
        mol = tensor_to_molecule(x[i], coords[i], bond_probs[i], config['model']['params']['max_nodes'])
        if mol:
            writer.write(mol)
            valid_molecules += 1
    writer.close()
    
    print(f"Generated {valid_molecules}/{num_samples} valid molecules")
    print(f"x shape: {x.shape}")
    print(f"coords shape: {coords.shape}")
    print(f"bond_probs shape: {bond_probs.shape}")

if __name__ == '__main__':
    generate()
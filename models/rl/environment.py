import gym
import numpy as np
from rdkit import Chem
from .reward import RewardFunction
from models.uedm.diffusion import UEDM
from scripts.generate import tensor_to_molecule
import torch

class MoleculeEnvironment(gym.Env):
    def __init__(self, config):
        super(MoleculeEnvironment, self).__init__()
        self.max_nodes = config.get('max_nodes', 50)
        self.max_steps = config.get('max_steps', 100)
        self.reward_function = RewardFunction(config)
        self.uedm_config = config.get('uedm', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация UEDM
        self.uedm = UEDM(self.uedm_config).to(self.device)
        checkpoint_path = 'checkpoints/uedm_epoch_10.pth'
        if os.path.exists(checkpoint_path):
            self.uedm.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded UEDM weights from {checkpoint_path}")
        
        self.observation_space = gym.spaces.Dict({
            'atom_types': gym.spaces.Box(low=0, high=100, shape=(self.max_nodes,), dtype=np.int32),
            'coords': gym.spaces.Box(low=-100, high=100, shape=(self.max_nodes, 3), dtype=np.float32),
            'bonds': gym.spaces.Box(low=0, high=4, shape=(self.max_nodes, self.max_nodes), dtype=np.int32)
        })
        
        self.action_space = gym.spaces.Discrete(5)
        
        self.mol = Chem.RWMol()
        self.step_count = 0
        self.pocket_x = None
        self.pocket_coords = None
    
    def reset(self, pocket_x=None, pocket_coords=None):
        self.mol = Chem.RWMol()
        self.step_count = 0
        
        # Используем UEDM для начальной генерации
        if pocket_x is not None and pocket_coords is not None:
            self.pocket_x = pocket_x.to(self.device)
            self.pocket_coords = pocket_coords.to(self.device)
            self.uedm.eval()
            with torch.no_grad():
                x, coords, bond_probs = self.uedm.sample(self.pocket_x, self.pocket_coords, 1, self.device, use_ddim=True, ddim_steps=50)
            mol = tensor_to_molecule(x[0], coords[0], bond_probs[0], self.max_nodes)
            if mol:
                self.mol = mol
        
        return self._get_observation()
    
    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        
        if action == 0:
            pass
        elif action == 1:
            self.mol.AddAtom(Chem.Atom('C'))
        elif action == 2 and self.mol.GetNumAtoms() > 0:
            self.mol.RemoveAtom(self.mol.GetNumAtoms() - 1)
        elif action == 3 and self.mol.GetNumAtoms() >= 2:
            i, j = np.random.choice(self.mol.GetNumAtoms(), 2, replace=False)
            self.mol.AddBond(i, j, Chem.BondType.SINGLE)
        elif action == 4 and self.mol.GetNumBonds() > 0:
            bonds = list(self.mol.GetBonds())
            bond = np.random.choice(bonds)
            self.mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        
        reward = self.reward_function.calculate_reward(self.mol, self.pocket_coords)
        obs = self._get_observation()
        return obs, reward, done, {}
    
    def _get_observation(self):
        atom_types = np.zeros(self.max_nodes, dtype=np.int32)
        coords = np.zeros((self.max_nodes, 3), dtype=np.float32)
        bonds = np.zeros((self.max_nodes, self.max_nodes), dtype=np.int32)
        
        for i, atom in enumerate(self.mol.GetAtoms()):
            atom_types[i] = atom.GetAtomicNum()
            conf = self.mol.GetConformer() if self.mol.GetNumConformers() > 0 else None
            if conf:
                coords[i] = conf.GetAtomPosition(i)
        
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}.get(bond.GetBondType(), 0)
            bonds[i, j] = bond_type
            bonds[j, i] = bond_type
        
        return {'atom_types': atom_types, 'coords': coords, 'bonds': bonds}
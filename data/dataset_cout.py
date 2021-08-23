#!/usr/bin/env/python
"""
Usage:
    make_dataset.py [options]

Options:
    -h --help        Show this screen.
    --file NAME      File
    --dataset NAME   Dataset
    --filter INT     Numero
"""

import json
import os
import sys

from docopt import docopt
from rdkit.Chem import rdmolops

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import utils
import seaborn as sns
import matplotlib.pyplot as plt
from model.datasets import *

# get current directory in order to work with full path and not dynamic
current_dir = os.path.dirname(os.path.realpath(__file__))


def readStr_qm9():
    f = open(current_dir + '/qm9.smi', 'r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    np.random.seed(1)
    np.random.shuffle(L)
    return L


def read_zinc():
    f = open(current_dir + '/zinc.smi', 'r')
    L = []
    for line in f:
        line = line.strip()
        L.append(line)
    f.close()
    return L


# add one edge to adj matrix
def add_edge_mat(amat, src, dest, e):
    amat[e, dest, src] = 1
    amat[e, src, dest] = 1


def graph_to_adj_mat(graph, max_n_vertices, num_edge_types):
    amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
    for src, e, dest in graph:
        add_edge_mat(amat, src, dest, e)
    return amat


def load_filter(data, number):
    res = []
    for n, s in enumerate(data):
        m = Chem.MolFromSmiles(s['smiles'])
        atoms = m.GetAtoms()
        if len(atoms) == number:
            res.append(s)
    return res


def load_data(file_name, number):
    print("Loading data from %s" % file_name)
    with open(file_name, 'r') as f:
        data = json.load(f)

    if number is not None:
        data = load_filter(data, int(number))

    return data


def count_molecules(data, dataset):
    if 'qm9' in str.lower(dataset):
        limit = 12
    else:
        limit = 40
    # groups molecules based on the number of atoms
    g_mol = [[] for i in range(limit)]
    for n, s in enumerate(data):
        m = Chem.MolFromSmiles(s['smiles'])
        atoms = m.GetAtoms()
        g_mol[len(atoms)].append(n)
    print('Grouped molecules: ', [(i, len(mols)) for i, mols in enumerate(g_mol)])
    n_g_mol = [len(mols) for mols in g_mol]
    sns.scatterplot(range(len(n_g_mol)), n_g_mol)
    plt.title("Number of molecules")
    plt.xlabel("Molecules by number of atoms")
    sns.despine(offset=True)
    plt.show()


def count_number_atoms_types(data, dataset):
    g_mol = np.zeros(len(utils.dataset_info(dataset)['atom_types']))
    # g_mol = {s:0 for s in utils.dataset_info(dataset)['atom_types']}
    for n, s in enumerate(data):
        m = Chem.MolFromSmiles(s['smiles'])
        atoms = m.GetAtoms()
        for atom in atoms:
            atom_str = utils.dataset_atom_rep(dataset, atom)
            if atom_str not in utils.dataset_info(dataset)['atom_types']:
                print('Unrecognized atom type %s' % atom_str)
                return None
            idx = utils.dataset_info(dataset)['atom_types'].index(atom_str)
            g_mol[idx] += 1
    atom_sum = sum(g_mol)
    print('Types of atoms: ', g_mol)
    if atom_sum > 0:
        print('Types of atoms %: ', [v / atom_sum for v in g_mol])
    else:
        print('Types of atoms %: ', 'division by 0')


def count_edges(data, dataset):
    num_fwd_edge_types = len(utils.dataset_info(dataset)['bond_types'])
    num_edge_types = num_fwd_edge_types
    edge_type = np.zeros(num_fwd_edge_types + 1)

    for s in data:
        n_atoms = len(s['node_features'])
        smiles = s['smiles']
        adj_mat = graph_to_adj_mat(s['graph'], n_atoms, num_edge_types)
        no_edge = 1 - np.sum(adj_mat, axis=0, keepdims=True)
        adj_mat = np.concatenate([no_edge, adj_mat], axis=0)
        for edge in range(num_fwd_edge_types + 1):
            tmp_sum = np.sum(adj_mat[edge, :, :])
            edge_type[edge] += tmp_sum

    print('Types of edges: ', edge_type)
    edge_sum = sum(edge_type)
    if edge_sum > 0:
        print('Types of edges %: ', [v / edge_sum for v in edge_type])
    else:
        print('Types of edges %: ', 'division by 0')


def upper_reconstruction(data, dataset, stamp=False):
    reconstruction = 0
    total = 0
    for s in data:
        # input data
        node_features = s['node_features']
        n_atoms = len(s['node_features'])
        # num_edge_types = len(utils.dataset_info(dataset)['bond_types'])
        num_edge_types = len(utils.bond_dict)
        adj_mat = graph_to_adj_mat(s['graph'], n_atoms, num_edge_types)
        edge_pred = np.sum(adj_mat, axis=0)
        edge_type_pred = adj_mat
        smiles = s['smiles']

        # Add atoms
        new_mol = Chem.MolFromSmiles('')
        new_mol = Chem.rdchem.RWMol(new_mol)
        node_features_one_hot = np.argmax(node_features, axis=-1)
        utils.add_atoms(new_mol, node_features_one_hot, dataset)
        valences = utils.get_initial_valence(node_features_one_hot, dataset)
        # Add edges
        for row in range(len(edge_pred[0])):
            for col in range(row + 1, len(edge_pred[0])):  # only half matrix
                if edge_pred[row, col] == True:
                    # choose an edge type
                    bond = np.argmax(edge_type_pred[:, row, col])
                    el = bond
                    if min(valences[row], valences[col]) >= el + 1:
                        new_mol.AddBond(int(row), int(col), utils.number_to_bond[el])
                        # utils.add_bonds(new_mol, bond, int(row), int(col), dataset)
                        valences[row] -= el + 1
                        valences[col] -= el + 1

        # Remove unconnected node
        utils.remove_extra_nodes(new_mol)
        # new_mol.UpdatePropertyCache(strict=False)  # needed for the following command
        # Chem.AssignStereochemistry(new_mol, force=True, cleanIt=False)  # fix properties
        pred_smiles = Chem.MolToSmiles(new_mol)
        new_mol = Chem.MolFromSmiles(pred_smiles)
        if new_mol is not None:
            if Chem.MolToSmiles(new_mol) == smiles:
                reconstruction = reconstruction + 1
            else:
                if stamp:
                    print('')
                    print(Chem.MolToSmiles(new_mol))
                    print(smiles)
        total = total + 1
        print("Dataset %s. Total: %i. Reconstruction: %.4f.  " % (dataset, total, reconstruction / total), end='\r')
    print("")


def try_smiles(smiles, dataset):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []

    # Kekulize it
    rdmolops.Kekulize(mol)
    if mol is None:
        return None, None
    print('asdasd', Chem.MolToSmiles(mol, kekuleSmiles=True))
    print('asdasd', Chem.MolToSmiles(mol, kekuleSmiles=False))
    edges = []
    nodes = []
    for bond in mol.GetBonds():
        bond_str = dataset_bond_rep(dataset, bond)
        if bond_str not in dataset_info(dataset)['bond_types']:
            print('Unrecognized bond type %s' % bond_str)
            return [], []
        bond_num = dataset_info(dataset)['bond_types'].index(bond_str)
        edges.append((bond.GetBeginAtomIdx(), bond_num, bond.GetEndAtomIdx()))
        print(bond_num, bond.GetStereo(), bond.GetBondDir(), bond.GetIsAromatic())

    for atom in mol.GetAtoms():
        atom_str = dataset_atom_rep(dataset, atom)
        print(atom_str, atom.GetChiralTag(), atom.GetIsAromatic())
        if atom_str not in dataset_info(dataset)['atom_types']:
            print('Unrecognized atom type %s' % atom_str)
            return [], []

        n_atoms = len(dataset_info(dataset)['atom_types'])
        nodes.append(utils.onehot(dataset_info(dataset)['atom_types'].index(atom_str), n_atoms))
    print('')

    # Add atoms
    n_atoms = len(nodes)
    num_edge_types = len(utils.dataset_info(dataset)['bond_types'])
    adj_mat = graph_to_adj_mat(edges, n_atoms, num_edge_types)
    edge_pred = np.sum(adj_mat, axis=0)
    edge_type_pred = adj_mat
    new_mol = Chem.MolFromSmiles('')
    new_mol = Chem.rdchem.RWMol(new_mol)
    node_features_one_hot = np.argmax(nodes, axis=-1)
    utils.add_atoms(new_mol, node_features_one_hot, dataset)
    valences = utils.get_initial_valence(node_features_one_hot, dataset)
    # Add edges
    for row in range(len(edge_pred[0])):
        for col in range(row + 1, len(edge_pred[0])):  # only half matrix
            if edge_pred[row, col] == True:
                # choose an edge type
                bond = np.argmax(edge_type_pred[:, row, col])
                el = dataset_info(dataset)['bond_el'][bond]
                if min(valences[row], valences[col]) >= el:
                    utils.add_bonds(new_mol, bond, int(row), int(col), dataset)
                    valences[row] -= el
                    valences[col] -= el

    # Remove unconnected node
    utils.remove_extra_nodes(new_mol)
    new_mol.UpdatePropertyCache(strict=False)  # needed for the following command
    Chem.AssignStereochemistry(new_mol, force=True, cleanIt=False)  # fix properties
    pred_smiles = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(pred_smiles)
    print("INPUT smiles :", smiles)
    print("OUTPUT smiles :", Chem.MolToSmiles(new_mol))


if __name__ == "__main__":
    args = docopt(__doc__)
    file = args.get('--file')
    dataset = args.get('--dataset')
    filter = args.get('--filter')

    data = load_data(file, filter)

    count_molecules(data, dataset)
    # count_number_atoms(data, dataset)
    # count_edges(data, dataset)
    # upper_reconstruction(data, dataset)
    # try_smiles('CCN(C[C@@H](C)/C(N)=N/O)C(=O)C1(C)CCCC1', dataset)
    # try_smiles('CNc1cnc[c+](C)o1', dataset)

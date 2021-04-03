#!/usr/bin/env/python
"""
Usage:
    make_dataset.py [options]

Options:
    -h --help        Show this screen.
    --file NAME      File
    --dataset NAME   Dataset
    --filter INT     Number
"""

import json
import os
import sys

from docopt import docopt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import utils
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


if __name__ == "__main__":
    args = docopt(__doc__)
    file = args.get('--file')
    dataset = args.get('--dataset')
    filter = args.get('--filter')
    data = load_data(file, filter)
    upper_reconstruction(data, dataset)

#!/usr/bin/env/python
import copy
import pickle
from collections import deque

from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from model.datasets import *

SMALL_NUMBER = 1e-7
LARGE_NUMBER = 1e10

geometry_numbers = [3, 4, 5, 6]  # triangle, square, pentagon, hexagon


# add one edge to adj matrix
def add_edge_mat(amat, src, dest, e, considering_edge_type=True):
    if considering_edge_type:
        amat[e, dest, src] = 1
        amat[e, src, dest] = 1
    else:
        amat[src, dest] = 1
        amat[dest, src] = 1


def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, considering_edge_type=True):
    if considering_edge_type:
        amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e)
    else:
        amat = np.zeros((max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e, considering_edge_type=False)
    return amat


# generates one hot vector
def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z


# standard normal with shape [a1, a2, a3]
def generate_std_normal(a1, a2, a3):
    return np.random.normal(0, 1, [a1, a2, a3])


# Get length for each graph based on node masks
def get_graph_length(all_node_mask):
    all_lengths = []
    for graph in all_node_mask:
        if 0 in graph:
            length = np.argmin(graph)
        else:
            length = len(graph)
        all_lengths.append(length)
    return all_lengths


# sample node symbols based on node predictions
def sample_node_symbol(all_node_symbol_prob, all_lengths, dataset):
    all_node_symbol = []
    for graph_idx, graph_prob in enumerate(all_node_symbol_prob):
        node_symbol = []
        for node_idx in range(all_lengths[graph_idx]):
            symbol = np.random.choice(np.arange(len(dataset_info(dataset)['atom_types'])), p=graph_prob[node_idx])
            node_symbol.append(symbol)
        all_node_symbol.append(node_symbol)
    return all_node_symbol


# sample node symbols based on node predictions
def sample_argmax_node_symbol(all_node_symbol_prob, all_lengths, dataset):
    all_node_symbol = []
    for graph_idx, graph_prob in enumerate(all_node_symbol_prob):
        node_symbol = []
        for node_idx in range(all_lengths[graph_idx]):
            symbol = np.arange(len(dataset_info(dataset)['atom_types']))[np.argmax(graph_prob[node_idx])]
            node_symbol.append(symbol)
        all_node_symbol.append(node_symbol)
    return all_node_symbol


# generate a new feature on whether adding the edges will generate more than two overlapped edges for rings
def get_overlapped_edge_feature(edge_mask, color, new_mol):
    overlapped_edge_feature = []
    for node_in_focus, neighbor in edge_mask:
        if color[neighbor] == 1:
            # attempt to add the edge
            new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[0])
            # Check whether there are two cycles having more than two overlap edges
            try:
                ssr = Chem.GetSymmSSSR(new_mol)  # smallest set of smallest rings
            except:
                ssr = []
            overlap_flag = False
            for idx1 in range(len(ssr)):
                for idx2 in range(idx1 + 1, len(ssr)):
                    if len(set(ssr[idx1]) & set(ssr[idx2])) > 2:
                        overlap_flag = True
            # remove that edge
            new_mol.RemoveBond(int(node_in_focus), int(neighbor))
            if overlap_flag:
                overlapped_edge_feature.append((node_in_focus, neighbor))
    return overlapped_edge_feature


# adj_list [3, v, v] or defaultdict. bfs distance on a graph
def bfs_distance(start, adj_list, is_dense=False):
    distances = {}
    visited = set()
    queue = deque([(start, 0)])
    visited.add(start)
    while len(queue) != 0:
        current, d = queue.popleft()
        for neighbor, edge_type in adj_list[current]:
            if neighbor not in visited:
                distances[neighbor] = d + 1
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
    return [(start, node, d) for node, d in distances.items()]


def get_initial_valence(node_symbol, dataset):
    return [dataset_info(dataset)['maximum_valence'][s] for s in node_symbol]


def get_idx_of_largest_frag(frags):
    return np.argmax([len(frag) for frag in frags])


def remove_extra_nodes(new_mol):
    frags = Chem.rdmolops.GetMolFrags(new_mol)
    while len(frags) > 1:
        # Get the idx of the frag with largest length
        largest_idx = get_idx_of_largest_frag(frags)
        for idx in range(len(frags)):
            if idx != largest_idx:
                # Remove one atom that is not in the largest frag
                new_mol.RemoveAtom(frags[idx][0])
                break
        frags = Chem.rdmolops.GetMolFrags(new_mol)


def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        print('converting radical electrons to H')
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False


def calc_node_mask(hist, dataset):
    mol_valence_list = []
    for key in dataset_info(dataset)['maximum_valence'].keys():
        mol_valence_list.append([dataset_info(dataset)['maximum_valence'][key]])
    H_b = np.greater(hist, 0)  # obtaining a vector of only 1 and 0
    idx = np.where(np.less_equal(H_b, 0))  # obtaining al the indexes with 0-values in the hist
    valences = np.add(idx, 1)  # valences to avoid
    equals = np.not_equal(mol_valence_list, valences)  # broadcasting.
    mask_bool = np.all(equals, 1)
    mask = mask_bool.astype(int).tolist()
    return mask


def incr_node(mol, dataset):
    hist_dim = dataset_info(dataset)['hist_dim']
    mol_valences = dataset_info(dataset)['maximum_valence']
    incr_hist = []
    incr_diff_hist = []
    incr_node_mask = []
    # calc increment hist
    c_hist = [0 for i in range(hist_dim)]
    incr_hist.append(np.copy(c_hist).tolist())
    incr_diff_hist.append(mol['hist'])
    incr_node_mask.append(calc_node_mask(mol['hist'], dataset))
    for n in mol['node_features']:
        idx_mol = np.argmax(n)  # n is a one hot representation
        val = mol_valences[idx_mol]
        val_idx = val - 1
        c_hist[val_idx] += 1
        incr_hist.append(np.copy(c_hist).tolist())

        # calc diff hist
        diff_hist = np.subtract(mol['hist'], c_hist)
        diff_hist = np.where(diff_hist > 0, diff_hist, np.zeros_like(diff_hist)).tolist()
        incr_diff_hist.append(diff_hist)

        # calc mask for the nodes_prob
        incr_node_mask.append(calc_node_mask(diff_hist, dataset))
    return incr_hist[:-1], incr_diff_hist[:-1], incr_node_mask[:-1]


def to_graph(smiles, dataset):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], []
    # Kekulize it
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None, None

    edges = []
    nodes = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
        assert bond_dict[str(bond.GetBondType())] != 3
    for atom in mol.GetAtoms():
        atom_str = dataset_atom_rep(dataset, atom)
        if atom_str not in dataset_info(dataset)['atom_types']:
            print('Unrecognized atom type %s' % atom_str)
            return [], []

        n_atoms = len(dataset_info(dataset)['atom_types'])
        nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), n_atoms))

    return nodes, edges


def shape_count(dataset, remove_print=False, all_smiles=None):
    if all_smiles == None:
        with open('generated_smiles_%s.txt' % dataset, 'rb') as f:
            all_smiles = set(pickle.load(f))

    geometry_counts = [0] * len(geometry_numbers)
    geometry_counts_per_molecule = []  # record the geometry counts for each molecule
    for smiles in all_smiles:
        nodes, edges = to_graph(smiles, dataset)
        if len(edges) <= 0:
            continue
        new_mol = Chem.MolFromSmiles(smiles)

        ssr = Chem.GetSymmSSSR(new_mol)
        counts_for_molecule = [0] * len(geometry_numbers)
        for idx in range(len(ssr)):
            ring_len = len(list(ssr[idx]))
            if ring_len in geometry_numbers:
                geometry_counts[geometry_numbers.index(ring_len)] += 1
                counts_for_molecule[geometry_numbers.index(ring_len)] += 1
        geometry_counts_per_molecule.append(counts_for_molecule)

    return len(all_smiles), geometry_counts, geometry_counts_per_molecule


def check_adjacent_sparse(adj_list, node, neighbor_in_doubt):
    for neighbor, edge_type in adj_list[node]:
        if neighbor == neighbor_in_doubt:
            return True, edge_type
    return False, None


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32)


# select the best based on shapes and probs
def select_best(all_mol):
    extracted = []
    for i in range(len(all_mol)):
        extracted.append((all_mol[i][0], all_mol[i][1], i))

    extracted = sorted(extracted)
    return all_mol[extracted[-1][2]][2]


# a series util function converting sparse matrix representation to dense
def incre_adj_mat_to_dense(incre_adj_mat, num_edge_types, maximum_vertice_num):
    new_incre_adj_mat = []
    for sparse_incre_adj_mat in incre_adj_mat:
        dense_incre_adj_mat = np.zeros((num_edge_types, maximum_vertice_num, maximum_vertice_num))
        for current, adj_list in sparse_incre_adj_mat.items():
            for neighbor, edge_type in adj_list:
                dense_incre_adj_mat[edge_type][current][neighbor] = 1
        new_incre_adj_mat.append(dense_incre_adj_mat)
    return new_incre_adj_mat  # [number_iteration,num_edge_types,maximum_vertice_num, maximum_vertice_num]


def distance_to_others_dense(distance_to_others, maximum_vertice_num):
    new_all_distance = []
    for sparse_distances in distance_to_others:
        dense_distances = np.zeros((maximum_vertice_num), dtype=int)
        for x, y, d in sparse_distances:
            dense_distances[y] = d
        new_all_distance.append(dense_distances)
    return new_all_distance  # [number_iteration, maximum_vertice_num]


def overlapped_edge_features_to_dense(overlapped_edge_features, maximum_vertice_num):
    new_overlapped_edge_features = []
    for sparse_overlapped_edge_features in overlapped_edge_features:
        dense_overlapped_edge_features = np.zeros((maximum_vertice_num), dtype=int)
        for node_in_focus, neighbor in sparse_overlapped_edge_features:
            dense_overlapped_edge_features[neighbor] = 1
        new_overlapped_edge_features.append(dense_overlapped_edge_features)
    return new_overlapped_edge_features  # [number_iteration, maximum_vertice_num]


def node_sequence_to_dense(node_sequence, maximum_vertice_num):
    new_node_sequence = []
    for node in node_sequence:
        s = [0] * maximum_vertice_num
        s[node] = 1
        new_node_sequence.append(s)
    return new_node_sequence  # [number_iteration, maximum_vertice_num]


def edge_type_masks_to_dense(edge_type_masks, maximum_vertice_num, num_edge_types):
    new_edge_type_masks = []
    for mask_sparse in edge_type_masks:
        mask_dense = np.zeros([num_edge_types, maximum_vertice_num])
        for node_in_focus, neighbor, bond in mask_sparse:
            mask_dense[bond][neighbor] = 1
        new_edge_type_masks.append(mask_dense)
    return new_edge_type_masks  # [number_iteration, 3, maximum_vertice_num]


def edge_type_labels_to_dense(edge_type_labels, maximum_vertice_num, num_edge_types):
    new_edge_type_labels = []
    for labels_sparse in edge_type_labels:
        labels_dense = np.zeros([num_edge_types, maximum_vertice_num])
        for node_in_focus, neighbor, bond in labels_sparse:
            labels_dense[bond][neighbor] = 1 / float(len(labels_sparse))  # fix the probability bug here.
        new_edge_type_labels.append(labels_dense)
    return new_edge_type_labels  # [number_iteration, 3, maximum_vertice_num]


def edge_masks_to_dense(edge_masks, maximum_vertice_num):
    new_edge_masks = []
    for mask_sparse in edge_masks:
        mask_dense = [0] * maximum_vertice_num
        for node_in_focus, neighbor in mask_sparse:
            mask_dense[neighbor] = 1
        new_edge_masks.append(mask_dense)
    return new_edge_masks  # [number_iteration, maximum_vertice_num]


def edge_labels_to_dense(edge_labels, maximum_vertice_num):
    new_edge_labels = []
    for label_sparse in edge_labels:
        label_dense = [0] * maximum_vertice_num
        for node_in_focus, neighbor in label_sparse:
            label_dense[neighbor] = 1 / float(len(label_sparse))
        new_edge_labels.append(label_dense)
    return new_edge_labels  # [number_iteration, maximum_vertice_num]

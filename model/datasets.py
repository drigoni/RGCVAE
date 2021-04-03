import numpy as np
from rdkit import Chem

# bond mapping
bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}
number_to_bond = {0: Chem.rdchem.BondType.SINGLE,
                  1: Chem.rdchem.BondType.DOUBLE,
                  2: Chem.rdchem.BondType.TRIPLE,
                  3: Chem.rdchem.BondType.AROMATIC}

bond_dir_dict = {'NONE': 0,
                 'BEGINWEDGE': 1,
                 'BEGINDASH': 2,
                 'ENDDOWNRIGHT': 3,
                 'ENDUPRIGHT': 4,
                 'EITHERDOUBLE': 5,
                 'UNKNOWN': 6}
number_to_bond_dir = {0: Chem.rdchem.BondDir.NONE,
                      1: Chem.rdchem.BondDir.BEGINWEDGE,
                      2: Chem.rdchem.BondDir.BEGINDASH,
                      3: Chem.rdchem.BondDir.ENDDOWNRIGHT,
                      4: Chem.rdchem.BondDir.ENDUPRIGHT,
                      5: Chem.rdchem.BondDir.EITHERDOUBLE,
                      6: Chem.rdchem.BondDir.UNKNOWN}

chi_dict = {'CHI_UNSPECIFIED': 0,
            'CHI_TETRAHEDRAL_CW': 1,
            'CHI_TETRAHEDRAL_CCW': 2,
            'CHI_OTHER': 3}
number_to_chi = {0: Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                 1: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                 2: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                 3: Chem.rdchem.ChiralType.CHI_OTHER}


def dataset_info(dataset):
    if dataset == 'qm9':
        values = {'atom_types': ["H", "C", "N", "O", "F"],
                  'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                  'hist_dim': 4,
                  'max_valence_value': 9,
                  'max_n_atoms': 30,
                  'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                  'bucket_sizes': np.array(list(range(4, 28, 2)) + [29]),
                  # 'n_edges': [6788282, 1883788, 222444, 63914],
                  # 'n_nodes': [0, 729895, 120522, 161809, 2828],
                  'n_edges': [3, 1, 1, 1],
                  'n_nodes': [1, 1, 1, 1, 1],
                  'batch_size': 100,
                  'n_epochs': 200,
                  }
    elif dataset == 'qm9_long2':
        values = {
            'atom_types': ["C4(0)0", "N3(0)0", "N2(-1)0", "O2(0)0", "F1(0)0", "C3(-1)0", "N4(1)0", "C4(1)0", "C3(1)0",
                           "O1(-1)0", "N3(1)0", "C2(0)0", "O3(1)0", "C4(0)1"],
            'maximum_valence': {0: 4, 1: 3, 2: 2, 3: 2, 4: 1, 5: 3, 6: 4, 7: 4, 8: 3,
                                9: 1, 10: 3, 11: 2, 12: 3, 13: 4},
            'hist_dim': 4,
            'max_valence_value': 9,
            'max_n_atoms': 30,
            'number_to_atom': {0: "C", 1: "N", 2: "N", 3: "O", 4: "F", 5: "C", 6: "N", 7: "C", 8: "C",
                               9: "O", 10: "N", 11: "C", 12: "O", 13: "C"},
            'bucket_sizes': np.array(list(range(4, 28, 2)) + [29]),
            # 'n_edges': [6788282, 1883788, 222444, 63914],
            # 'n_nodes': [725365, 93611, 7903, 161513, 2828, 705, 19003, 443, 3377,
            #             295, 5, 1, 1, 4],
            'n_edges': [3, 1, 1, 1],
            'n_nodes': [1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1],
            'batch_size': 100,
            'n_epochs': 200,
        }
    elif dataset == 'zinc':
        values = {'atom_types': ["H", "C", "N", "O", "F", "S", "Cl", "Br", "I"],
                  'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1, 5: 6, 6: 7, 7: 5, 8: 7},
                  'hist_dim': 7,
                  'max_valence_value': 34,
                  'max_n_atoms': 85,
                  'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F", 5: "S", 6: "Cl", 7: "Br", 8: "I"},
                  'bucket_sizes': np.array(
                      [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58,
                       84]),
                  # 'n_edges': [111623688, 8153922, 2791900, 27394],
                  # 'n_nodes': [0, 3764414, 624130, 509483, 70097, 90962, 90962, 11220, 800],
                  'n_edges': [3, 1, 1, 1],
                  'n_nodes': [1, 1, 1, 1, 1, 1, 1, 1, 1],
                  'batch_size': 100,
                  'n_epochs': 70,
                  }
    elif dataset == 'zinc_long2':
        values = {
            'atom_types': ['Br1(0)0', 'C4(0)0', 'Cl1(0)0', 'F1(0)0', 'H1(0)0', 'I1(0)0', 'N2(-1)0', 'N3(0)0', 'N4(1)0',
                           'O1(-1)0',
                           'O2(0)0', 'S2(0)0', 'S4(0)0', 'S6(0)0', "C4(0)1", "C4(0)2", 'S4(0)2', 'S1(-1)0', 'S4(0)1',
                           "O3(1)0", 'S6(0)2', "P5(0)0", "P5(0)1", "P4(1)0", "S3(1)0", "C3(-1)0", "P5(0)2", "P3(0)0",
                           "S6(0)1", "S3(1)1"],
            'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1,
                                10: 2, 11: 2, 12: 4, 13: 6, 14: 4, 15: 4, 16: 4, 17: 1, 18: 4,
                                19: 3, 20: 6, 21: 5, 22: 5, 23: 4, 24: 3, 25: 3, 26: 5, 27: 3,
                                28: 6, 29: 3},
            'hist_dim': 6,
            'max_valence_value': 34,
            'max_n_atoms': 85,
            'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                               10: 'O', 11: 'S', 12: 'S', 13: 'S', 14: "C", 15: "C", 16: "S", 17: "S", 18: "S",
                               19: "O", 20: "S", 21: "P", 22: "P", 23: "P", 24: "S", 25: "C", 26: "P", 27: "P",
                               28: "S", 29: "S"},
            'bucket_sizes': np.array(
                [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58,
                 84]),
            # 'n_edges': [111623688, 8153922, 2791900, 27394],
            # 'n_nodes': [11233, 3570490, 37961, 70252, 0, 785, 1363, 555064, 68066, 21567,
            #             488317, 63847, 80, 24651, 94566, 100218, 930, 392, 955,
            #             16, 3, 55, 27, 2, 5, 3, 22, 4,
            #             5, 1],
            'n_edges': [3, 1, 1, 1],
            'n_nodes': [1] * 30,
            'batch_size': 100,
            'n_epochs': 70,
        }
    else:
        print("Error: The datasets that you could use are QM9 or ZINC, not " + str(dataset))
        exit(1)
    # loss existing edges
    edges_to_consider = values['n_edges'][1:]  # the first position represent the non-present edges
    n_no_edges = values['n_edges'][0]
    n_yes_edges = sum(edges_to_consider)
    n_no_edges = max(n_no_edges, n_yes_edges) / n_no_edges
    n_yes_edges = max(n_no_edges, n_yes_edges) / n_yes_edges
    values['loss_no_edge'] = n_no_edges / max(n_no_edges, n_yes_edges)
    values['loss_yes_edge'] = n_yes_edges / max(n_no_edges, n_yes_edges)

    # values not normalized
    values['loss_node_weights'] = [max(values['n_nodes']) / i if i > 0 else 1
                                   for i in values['n_nodes']]
    # normalized values
    values['loss_node_weights'] = [i / max(values['loss_node_weights'])
                                   for i in values['loss_node_weights']]
    # values not normalized
    values['loss_edge_weights'] = [max(edges_to_consider) / i if i > 0 else 1
                                   for i in edges_to_consider]
    # normalized values
    values['loss_edge_weights'] = [i / max(values['loss_edge_weights'])
                                   for i in values['loss_edge_weights']]
    return values


def dataset_atom_rep(dataset, atom):
    if dataset == 'qm9' or dataset == 'zinc':
        atom_str = atom.GetSymbol()
    elif dataset == 'qm9_long2' or dataset == 'zinc_long2':
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        chi = atom.GetChiralTag()
        atom_str = "%s%i(%i)%i" % (symbol, valence, charge, chi)
    else:
        print('Unrecognized dataset')
        exit(1)
    return atom_str


def add_atoms(new_mol, node_symbol, dataset):
    for number in node_symbol:
        if dataset == 'qm9' or dataset == 'zinc':
            new_mol.AddAtom(Chem.Atom(dataset_info(dataset)['number_to_atom'][number]))
        elif dataset == 'qm9_long2' or dataset == 'zinc_long2':
            new_atom = Chem.Atom(dataset_info(dataset)['number_to_atom'][number])
            charge_num = int(dataset_info(dataset)['atom_types'][number][:-1].split('(')[1].strip(')'))
            chi_number = int(dataset_info(dataset)['atom_types'][number][-1])
            new_atom.SetFormalCharge(charge_num)
            new_atom.SetChiralTag(number_to_chi[chi_number])
            new_mol.AddAtom(new_atom)
        else:
            print('Unrecognized dataset')
            exit(1)


def add_bonds(new_mol, bond, row, col, dataset):
    bond_str = number_to_bond[bond][:-1]
    bond_prop = number_to_bond[bond][-1]
    new_mol.AddBond(int(row), int(col), bond_str)

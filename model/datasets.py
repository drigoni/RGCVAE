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
    elif dataset == 'qm9_ev':
        values = {'atom_types': ["C1", "N1", "N4", "F1", "N3", "C3", "C4", "O2", "O4",
                                 "N2", "C2", "O1", "O3"],
                  'maximum_valence': {0: 1, 1: 1, 2: 4, 3: 1, 4: 3, 5: 3, 6: 4, 7: 2, 8: 4,
                                      9: 2, 10: 2, 11: 1, 12: 3},
                  'hist_dim': 4,
                  'max_valence_value': 9,
                  'max_n_atoms': 30,
                  'number_to_atom': {0: "C", 1: "N", 2: "N", 3: "F", 4: "N", 5: "C", 6: "C", 7: "O", 8: "O",
                                     9: "N", 10: "C", 11: "O", 12: "O"},
                  'bucket_sizes': np.array(list(range(4, 28, 2)) + [29]),
                  'n_edges': [6788282, 1883788, 222444, 63914],
                  'n_nodes': [99645, 11763, 19003, 2828, 55743, 265029, 166614, 116325, 0,
                              34013, 198607, 45483, 1],
                  'batch_size': 100,
                  'n_epochs': 200,
                  }
    elif dataset == 'qm9_ev2':
        values = {'atom_types': ["C1(0)", "N1(0)", "N4(1)", "F1(0)", "N3(0)", "C3(0)", "C4(0)", "O2(0)",
                                 "N2(0)", "C2(0)", "O1(0)", "N2(-1)", "C4(1)", "C3(1)", "C3(-1)", "O1(-1)",
                                 "N3(1)", "O3(1)"],
                  'maximum_valence': {0: 1, 1: 1, 2: 4, 3: 1, 4: 3, 5: 3, 6: 4, 7: 2,
                                      8: 2, 9: 2, 10: 1, 11: 2, 12: 4, 13: 3, 14: 3, 15: 1,
                                      16: 1, 17: 3},
                  'hist_dim': 4,
                  'max_valence_value': 9,
                  'max_n_atoms': 30,
                  'number_to_atom': {0: "C", 1: "N", 2: "N", 3: "F", 4: "N", 5: "C", 6: "C", 7: "O",
                                     8: "N", 9: "C", 10: "O", 11: "N", 12: "C", 13: "C", 14: "C", 15: "O",
                                     16: "N", 17: "O"},
                  'bucket_sizes': np.array(list(range(4, 28, 2)) + [29]),
                  'n_edges': [6788282, 1883788, 222444, 63914],
                  'n_nodes': [99645, 11763, 19003, 2828, 55738, 260947, 166171, 116325,
                              26110, 198607, 45188, 7903, 443, 337, 705, 295,
                              5, 1],
                  'batch_size': 100,
                  'n_epochs': 200,
                  }
    elif dataset == 'qm9_long':
        values = {'atom_types': ["C4(0)", "N3(0)", "N2(-1)", "O2(0)", "F1(0)", "C3(-1)", "N4(1)", "C4(1)", "C3(1)",
                                 "O1(-1)", "N3(1)", "C2(0)", "O3(1)"],
                  'maximum_valence': {0: 4, 1: 3, 2: 2, 3: 2, 4: 1, 5: 3, 6: 4, 7: 4, 8: 3,
                                      9: 1, 10: 3, 11: 2, 12: 3},
                  'hist_dim': 4,
                  'max_valence_value': 9,
                  'max_n_atoms': 30,
                  'number_to_atom': {0: "C", 1: "N", 2: "N", 3: "O", 4: "F", 5: "C", 6: "N", 7: "C", 8: "C",
                                     9: "O", 10: "N", 11: "C", 12: "O"},
                  'bucket_sizes': np.array(list(range(4, 28, 2)) + [29]),
                  'n_edges': [6788282, 1883788, 222444, 63914],
                  'n_nodes': [725369, 93611, 7903, 161513, 2828, 705, 19003, 443, 3377,
                              295, 5, 1, 1],
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
                  'n_epochs': 200,
                  }
    elif dataset == 'zinc_ev':
        values = {'atom_types': ["C1", "N1", "N4", "F1", "N3", "C3", "C4", "O2", "O4", "N2", "C2", "O1",
                                 "O3", "S6", "S2", "Br1", "Cl1", "S4", "I1",
                                 "S1", "S3"],
                  'maximum_valence': {0: 1, 1: 1, 2: 4, 3: 1, 4: 3, 5: 3, 6: 4, 7: 2, 8: 4, 9: 2, 10: 2, 11: 1,
                                      12: 3, 13: 6, 14: 2, 15: 1, 16: 1, 17: 4, 18: 1,
                                      19: 1, 20: 3},
                  'hist_dim': 6,
                  'max_valence_value': 34,  # used in hist
                  'max_n_atoms': 85,
                  'number_to_atom': {0: "C", 1: "N", 2: "N", 3: "F", 4: "N", 5: "C", 6: "C", 7: "O", 8: "O",
                                     9: "N", 10: "C", 11: "O", 12: "O", 13: "S", 14: "S", 15: "Br", 16: "Cl", 17: "S",
                                     18: "I",
                                     19: "S", 20: "S"},
                  'bucket_sizes': np.array(
                      [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58,
                       84]),
                  'n_edges': [6788282, 1883788, 222444, 63914],
                  'n_nodes': [390455, 15355, 68143, 70045, 376805, 1212178, 1327055, 461739, 0, 163746, 834636, 47802,
                              16, 24669, 63488, 11224, 37885, 1995, 783,
                              805, 5],
                  'batch_size': 100,
                  'n_epochs': 200,
                  }
    elif dataset == 'zinc_ev2':
        values = {'atom_types': ["C1(0)", "N1(0)", "N4(1)", "F1(0)", "N3(0)", "C3(0)", "C4(0)", "O2(0)",
                                 "N2(0)", "C2(0)", "O1(0)", "O3(1)", "S6(0)", "S2(0)", "Br1(0)", "Cl1(0)",
                                 "S4(0)", "I1(0)", "S1(0)", "S3(1)", "O1(-1)", "N2(-1)", "S1(-1)", "C3(-1)"],
                  'maximum_valence': {0: 1, 1: 1, 2: 4, 3: 1, 4: 3, 5: 3, 6: 4, 7: 2,
                                      8: 2, 9: 2, 10: 1, 11: 3, 12: 6, 13: 2, 14: 1, 15: 1,
                                      16: 4, 17: 1, 18: 1, 19: 3, 20: 1, 21: 2, 22: 1, 23: 3},
                  'hist_dim': 6,
                  'max_valence_value': 34,  # used in hist
                  'max_n_atoms': 85,
                  'number_to_atom': {0: "C", 1: "N", 2: "N", 3: "F", 4: "N", 5: "C", 6: "C", 7: "O",
                                     8: "N", 9: "C", 10: "O", 11: "O", 12: "S", 13: "S", 14: "Br", 15: "Cl", 16: "S",
                                     17: "I",
                                     18: "S", 19: "S", 20: "O", 21: "N", 22: "S", 23: "C"},
                  'bucket_sizes': np.array(
                      [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58,
                       84]),
                  'n_edges': [6788282, 1883788, 222444, 63914],
                  'n_nodes': [389601, 15334, 68041, 70003, 376904, 1212826, 1326480, 461336,
                              162405, 834993, 26444, 13, 24696, 63532, 11265, 37927,
                              1983, 803, 420, 6, 21380, 1362, 403, 3],
                  'batch_size': 100,
                  'n_epochs': 200,
                  }
    elif dataset == 'zinc_long':
        values = {
            'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)', 'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)',
                           'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'],
            'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1,
                                10: 2, 11: 2, 12: 4, 13: 6},
            'hist_dim': 6,
            'max_valence_value': 34,
            'max_n_atoms': 85,
            'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                               10: 'O', 11: 'S', 12: 'S', 13: 'S'},
            'bucket_sizes': np.array(
                [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58,
                 84]),
            'n_edges': [111623688, 8153922, 2791900, 27394],
            'n_nodes': [11251, 3758488, 37721, 70303, 0, 799, 1337, 553583, 67890, 21442,
                        487609, 63815, 1980, 24630],
            'batch_size': 100,
            'n_epochs': 200,
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
            # 'bucket_sizes': np.array(list(range(10, 40, 2))),
            # 'n_edges': [111623688, 8153922, 2791900, 27394],
            # 'n_nodes': [11233, 3570490, 37961, 70252, 0, 785, 1363, 555064, 68066, 21567,
            #             488317, 63847, 80, 24651, 94566, 100218, 930, 392, 955,
            #             16, 3, 55, 27, 2, 5, 3, 22, 4,
            #             5, 1],
            'n_edges': [3, 1, 1, 1],
            'n_nodes': [1] * 30,
            'batch_size': 100,
            'n_epochs': 200,
        }
    elif dataset == 'zinc1M_long2':
        values = {
            'atom_types': ['Cl1(0)0', 'S2(0)0', 'N3(0)0', 'O2(0)0', 'F1(0)0', 'C4(0)0', 'S6(0)0', 'S4(0)0', 'Br1(0)0'],
            'maximum_valence': {0: 1, 1: 2, 2: 3, 3: 2, 4: 1, 5: 4, 6: 6, 7: 4, 8: 1},
            'hist_dim': 6,
            'max_valence_value': 34,
            'max_n_atoms': 85,
            'number_to_atom': {0: 'Cl', 1: 'S', 2: 'N', 3: 'O', 4: 'F', 5: 'C', 6: 'S', 7: 'S', 8: 'B'},
            'bucket_sizes': np.array(
                [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58,
                 84]),
            # 'n_edges': [6.06030803e+08, 4.75575800e+07, 1.80507840e+07, 2.27430000e+05],
            # 'n_nodes':  [1.6752500e+05, 3.4812500e+05, 4.1872520e+06, 3.1869590e+06, 4.4005500e+05, 2.2158453e+07, 1.5217000e+05, 2.7260000e+03, 4.7220000e+04],
            'n_edges': [3, 1, 1, 1],
            'n_nodes': [1] * 9,
            'batch_size': 1024,

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
    # values['loss_no_edge'] = 1
    # values['loss_yes_edge'] = n_no_edges / n_yes_edges

    # values not normalized
    values['loss_node_weights'] = [max(values['n_nodes']) / i if i > 0 else 1
                                   for i in values['n_nodes']]
    # normalized values
    values['loss_node_weights'] = [i / max(values['loss_node_weights'])
                                   for i in values['loss_node_weights']]
    # values['loss_node_weights'] = [1 if i > 0 else 1
    #                                for i in values['n_nodes']]
    # values not normalized
    values['loss_edge_weights'] = [max(edges_to_consider) / i if i > 0 else 1
                                   for i in edges_to_consider]
    # normalized values
    values['loss_edge_weights'] = [i / max(values['loss_edge_weights'])
                                   for i in values['loss_edge_weights']]
    # values['loss_edge_weights'] = [1 if i > 0 else 1
    #                                for i in edges_to_consider]
    return values


def dataset_atom_rep(dataset, atom):
    if dataset == 'qm9' or dataset == 'zinc':
        atom_str = atom.GetSymbol()
    elif dataset == 'qm9_ev' or dataset == 'zinc_ev':
        symbol = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        atom_str = "%s%i" % (symbol, valence)
    elif dataset == 'qm9_ev2' or dataset == 'zinc_ev2':
        symbol = atom.GetSymbol()
        valence = atom.GetExplicitValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)
    elif dataset == 'qm9_long' or dataset == 'zinc_long':
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)
    elif dataset == 'qm9_long2' or dataset == 'zinc_long2':
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        chi = atom.GetChiralTag()
        atom_str = "%s%i(%i)%i" % (symbol, valence, charge, chi)
    elif dataset == 'zinc1M_long2':
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
        if dataset == 'qm9' or dataset == 'zinc' or dataset == 'qm9_ev' or dataset == 'zinc_ev':
            new_mol.AddAtom(Chem.Atom(dataset_info(dataset)['number_to_atom'][number]))
        elif dataset == 'qm9_ev2' or dataset == 'zinc_ev2':
            new_atom = Chem.Atom(dataset_info(dataset)['number_to_atom'][number])
            charge_num = int(dataset_info(dataset)['atom_types'][number].split('(')[1].strip(')'))
            new_atom.SetFormalCharge(charge_num)
            new_mol.AddAtom(new_atom)
        elif dataset == 'qm9_long' or dataset == 'zinc_long':
            new_atom = Chem.Atom(dataset_info(dataset)['number_to_atom'][number])
            charge_num = int(dataset_info(dataset)['atom_types'][number].split('(')[1].strip(')'))
            new_atom.SetFormalCharge(charge_num)
            new_mol.AddAtom(new_atom)
        elif dataset == 'qm9_long2' or dataset == 'zinc_long2':
            new_atom = Chem.Atom(dataset_info(dataset)['number_to_atom'][number])
            charge_num = int(dataset_info(dataset)['atom_types'][number][:-1].split('(')[1].strip(')'))
            chi_number = int(dataset_info(dataset)['atom_types'][number][-1])
            new_atom.SetFormalCharge(charge_num)
            new_atom.SetChiralTag(number_to_chi[chi_number])
            new_mol.AddAtom(new_atom)
        elif dataset == 'zinc1M_long2':
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

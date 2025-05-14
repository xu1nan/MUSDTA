# Adapted from https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py

import pickle
import numpy as np
from openbabel import pybel

def CusBondFeaturizer(bond):
    return [int(bond.IsAromatic()), int(bond.IsInRing())]



class Featurizer():
    """
        Computes atomic features for molecules. Features can encode atom types,
        native PyBel properties or any custom properties defined using SMARTS patterns.

        Attributes
        ----------
        FEATURE_NAMES: list of str
            Labels for features (in the same order as features)
        NUM_ATOM_CLASSES: int
            Number of atom codes
        ATOM_CODES: dict
            Dictionary mapping atomic numbers to codes
        NAMED_PROPS: list of str
            Names of atom properties to retrieve from pybel.Atom objects
        CALLABLES: list of callables
            Callables to compute custom atomic properties
        SMARTS: list of SMARTS strings
            SMARTS patterns defining additional atomic properties
        """

    def __init__(self, atom_codes=None, atom_labels=None,
                 named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None,
                 smarts_labels=None):
        """
               Create featurizer with specified feature types. Feature vector elements will be ordered as:
               atom type encoding (defined by atom_codes), Pybel atom properties (defined by named_properties),
               molecule codes (if present), custom atom properties (defined by custom_properties),
               and additional properties defined using SMARTS (defined by smarts_properties).

               Parameters
               ----------
               atom_codes: dict, optional
                   Dictionary mapping atomic numbers to codes. Used for one-hot encoding.
                   Codes should range from 0 to n-1 for n different types.
                   Multiple atoms can share same code (e.g. {6:0, 7:1, 8:1} encodes carbon as [1,0]
                   and nitrogen/oxygen as [0,1]). If None, default encoding is used.
               atom_labels: list of str, optional
                   Labels for atom codes. Should have same length as number of codes used.
                   If None, labels like 'atom0', 'atom1' are used.
                   Ignored if atom_codes is None.
               named_properties: list of str, optional
                   Names of atom properties to retrieve from pybel.Atom objects.
                   If None, defaults to ['hyb', 'heavyvalence', 'heterovalence', 'partialcharge'].
               save_molecule_codes: bool, optional (default True)
                   If True, includes additional feature for molecule type code.
                   Useful when saving molecular complexes in single array.
               custom_properties: list of callables, optional
                   Custom functions to compute atomic properties.
                   Each element should be callable taking pybel.Atom and returning float.
                   If callable has __name__ attribute, it's used as feature label.
                   Otherwise uses labels like 'func<i>' where i is index in custom_properties list.
               smarts_properties: list of str, optional
                   Additional atomic properties defined using SMARTS patterns.
                   Patterns should match single atoms. If None, default patterns are used.
               smarts_labels: list of str, optional
                   Labels for SMARTS-defined properties. Should have same length as smarts_properties.
                   If None, uses labels like 'smarts0', 'smarts1'.
                   Ignored if smarts_properties is None.
               """
        # Remember names of all features in correct order
        self.FEATURE_NAMES = []

        if atom_codes is not None:
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values())
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)

            self.NUM_ATOM_CLASSES = len(codes)
            self.ATOM_CODES = atom_codes
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: '
                                     '%s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels
        else:
            self.ATOM_CODES = {}

            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

            # List of tuples (atomic_num, class_name) with atom types to encode.
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]

            for code, (atom, name) in enumerate(atom_classes):
                if type(atom) is list:
                    #
                    for a in atom:
                        self.ATOM_CODES[a] = code
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name)

            self.NUM_ATOM_CLASSES = len(atom_classes)

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom)
                             if not prop.startswith('__')]
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError(
                        'named_properties must be in pybel.Atom attributes,'
                        ' %s was given at position %s' % (prop_id, prop)
                    )
            self.NAMED_PROPS = named_properties
        else:
            # pybel.Atom properties to save
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS

        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s '
                            'instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:
            # Remember if an atom belongs to the ligand or to the protein
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:
            for i, func in enumerate(custom_properties):
                if not callable(func):
                    raise TypeError('custom_properties should be list of'
                                    ' callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            # SMARTS definition for other properties
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s'
                                 ' instead of %s'
                                 % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        # Compile patterns
        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        """Encode atom type with a binary vector. If atom type is not included in
        the `atom_classes`, its encoding is an all-zeros vector.

        Parameters
        ----------
        atomic_num: int
            Atomic number

        Returns
        -------
        encoding: np.ndarray
            Binary vector encoding atom type (one-hot or null).
        """

        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given'
                            % type(atomic_num))

        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def find_smarts(self, molecule):
        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_features(self, molecule, molcode=None):
        """
            Get coordinates and features for all heavy atoms in molecule.
            Parameters
               ----------
            molecule: pybel.Molecule
                Molecule to process
            molcode: float, optional  Molecule type code. Can be used to encode whether atom belongs toligand (1.0) or protein (-1.0) etc.

            Returns
               -------
            coords: np.ndarray, shape = (N, 3) Coordinates of all heavy atoms in molecule
            features: np.ndarray, shape = (N, F) Features of all heavy atoms: atom type (one-hot encoded),
            pybel.Atom properties, molecule type code (if enabled),and additional SMARTS-defined properties
        """
        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords = []
        features = []
        heavy_atoms = []
        h_num = []
        num = 0

        for i, atom in enumerate(molecule):
            if atom.atomicnum == 1:
                h_num.append(num)
                num += 1
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            elif atom.atomicnum > 1:
                h_num.append(num)
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))

        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1))))
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features, h_num

    def get_features_CSAR(self, molecule, protein_idxs, ligand_idxs, molcode=None):
        """Get coordinates and features for all heavy atoms in the molecule.

        Parameters
        ----------
        molecule: pybel.Molecule
        molcode: float, optional
            Molecule type. You can use it to encode whether an atom belongs to
            the ligand (1.0) or to the protein (-1.0) etc.

        Returns
        -------
        coords: np.ndarray, shape = (N, 3)
            Coordinates of all heavy atoms in the `molecule`.
        features: np.ndarray, shape = (N, F)
            Features of all heavy atoms in the `molecule`: atom type
            (one-hot encoding), pybel.Atom attributes, type of a molecule
            (e.g protein/ligand distinction), and other properties defined with
            SMARTS patterns
        """

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object,'
                            ' %s was given' % type(molecule))
        if molcode is None:
            if self.save_molecule_codes is True:
                raise ValueError('save_molecule_codes is set to True,'
                                 ' you must specify code for the molecule')
        elif not isinstance(molcode, (float, int)):
            raise TypeError('motlype must be float, %s was given'
                            % type(molcode))

        coords,protein_coords,ligand_coords = [],[],[]
        features,protein_features,ligand_features = [],[],[]
        heavy_atoms,protein_heavy_atoms,ligand_heavy_atoms = [],[],[]

        for i, atom in enumerate(molecule):
            # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
            index = i
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)

                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))
                if index in protein_idxs:
                    protein_heavy_atoms.append(i)
                    protein_coords.append(atom.coords)
                    protein_features.append(np.concatenate((
                        self.encode_num(atom.atomicnum),
                        [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                        [func(atom) for func in self.CALLABLES],
                    )))
                elif index in ligand_idxs:
                    ligand_heavy_atoms.append(i)
                    ligand_coords.append(atom.coords)
                    ligand_features.append(np.concatenate((
                        self.encode_num(atom.atomicnum),
                        [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                        [func(atom) for func in self.CALLABLES],
                    )))

        coords,protein_coords,ligand_coords = np.array(coords, dtype=np.float32),\
                                              np.array(protein_coords, dtype=np.float32),\
                                              np.array(ligand_coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features,
                                  molcode * np.ones((len(features), 1))))
        features = np.hstack([features,
                              self.find_smarts(molecule)[heavy_atoms]])
        protein_features = np.hstack([protein_features,
                              self.find_smarts(molecule)[protein_heavy_atoms]])
        ligand_features = np.hstack([ligand_features,
                              self.find_smarts(molecule)[ligand_heavy_atoms]])

        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')

        return coords, features, protein_coords, protein_features, ligand_coords, ligand_features

    def to_pickle(self, fname='featurizer.pkl'):
        """Save featurizer in a given file. Featurizer can be restored with
        `from_pickle` method.

        Parameters
        ----------
        fname: str, optional
           Path to file in which featurizer will be saved
        """

        # patterns can't be pickled, we need to temporarily remove them
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname):
        """Load pickled featurizer from a given file

        Parameters
        ----------
        fname: str, optional
           Path to file with saved featurizer

        Returns
        -------
        featurizer: Featurizer object
           Loaded featurizer
        """
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer
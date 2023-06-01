import argparse, copy, gzip, re, sys
from pathlib import Path
from typing import Any, Iterator, List, Optional, Protocol, Set, Tuple, Union
from calculate_rmsd import * 
import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial import distance_matrix  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore

try:
    import qml  # type: ignore
except ImportError:  # pragma: no cover
    qml = None  # pragma: no cover


METHOD_KABSCH = "kabsch"
METHOD_QUATERNION = "quaternion"
METHOD_NOROTATION = "none"
ROTATION_METHODS = [METHOD_KABSCH, METHOD_QUATERNION, METHOD_NOROTATION]


REORDER_NONE = "none"
REORDER_QML = "qml"
REORDER_HUNGARIAN = "hungarian"
REORDER_INERTIA_HUNGARIAN = "inertia-hungarian"
REORDER_BRUTE = "brute"
REORDER_DISTANCE = "distance"
REORDER_METHODS = [
    REORDER_NONE,
    REORDER_QML,
    REORDER_HUNGARIAN,
    REORDER_INERTIA_HUNGARIAN,
    REORDER_BRUTE,
    REORDER_DISTANCE,
]


AXIS_SWAPS = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]])

AXIS_REFLECTIONS = np.array(
    [
        [1, 1, 1],
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
    ]
)

ELEMENT_WEIGHTS = {
    1: 1.00797,
    2: 4.00260,
    3: 6.941,
    4: 9.01218,
    5: 10.81,
    6: 12.011,
    7: 14.0067,
    8: 15.9994,
    9: 18.998403,
    10: 20.179,
    11: 22.98977,
    12: 24.305,
    13: 26.98154,
    14: 28.0855,
    15: 30.97376,
    16: 32.06,
    17: 35.453,
    19: 39.0983,
    18: 39.948,
    20: 40.08,
    21: 44.9559,
    22: 47.90,
    23: 50.9415,
    24: 51.996,
    25: 54.9380,
    26: 55.847,
    28: 58.70,
    27: 58.9332,
    29: 63.546,
    30: 65.38,
    31: 69.72,
    32: 72.59,
    33: 74.9216,
    34: 78.96,
    35: 79.904,
    36: 83.80,
    37: 85.4678,
    38: 87.62,
    39: 88.9059,
    40: 91.22,
    41: 92.9064,
    42: 95.94,
    43: 98,
    44: 101.07,
    45: 102.9055,
    46: 106.4,
    47: 107.868,
    48: 112.41,
    49: 114.82,
    50: 118.69,
    51: 121.75,
    53: 126.9045,
    52: 127.60,
    54: 131.30,
    55: 132.9054,
    56: 137.33,
    57: 138.9055,
    58: 140.12,
    59: 140.9077,
    60: 144.24,
    61: 145,
    62: 150.4,
    63: 151.96,
    64: 157.25,
    65: 158.9254,
    66: 162.50,
    67: 164.9304,
    68: 167.26,
    69: 168.9342,
    70: 173.04,
    71: 174.967,
    72: 178.49,
    73: 180.9479,
    74: 183.85,
    75: 186.207,
    76: 190.2,
    77: 192.22,
    78: 195.09,
    79: 196.9665,
    80: 200.59,
    81: 204.37,
    82: 207.2,
    83: 208.9804,
    84: 209,
    85: 210,
    86: 222,
    87: 223,
    88: 226.0254,
    89: 227.0278,
    91: 231.0359,
    90: 232.0381,
    93: 237.0482,
    92: 238.029,
    94: 242,
    95: 243,
    97: 247,
    96: 247,
    102: 250,
    98: 251,
    99: 252,
    108: 255,
    109: 256,
    100: 257,
    101: 258,
    103: 260,
    104: 261,
    107: 262,
    105: 262,
    106: 263,
    110: 269,
    111: 272,
    112: 277,
}

ELEMENT_NAMES = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    114: "Uuq",
    116: "Uuh",
}

NAMES_ELEMENT = {value: key for key, value in ELEMENT_NAMES.items()}

def get_rmsd(p_all_atoms, p_all, q_all_atoms, q_all, reorder = True,
             reorder_method = 'brute', rotation_method = 'kabsch',
             use_reflections = False, use_ref_stereo = False,
             output = True) -> None:

    """

    p/q_all_atoms -> list of atomic numbers
    p/q_all -> np.ndarray of coordinates (N,3)

    reorder_method -> 'brute', 'hungarian'
    reorder_method -> 'kabsch', 'quaternion'

    """

    # Parse arguments
    settings = parse_arguments(args)

    p_size = p_all.shape[0]
    q_size = q_all.shape[0]

    if not p_size == q_size:
        raise ValueError("error: Structures not same size")

    if np.count_nonzero(p_all_atoms != q_all_atoms) and not settings.reorder:
        msg = """
error: Atoms are not in the same order.

Use --reorder to align the atoms (can be expensive for large structures).

Please see --help or documentation for more information or
https://github.com/charnley/rmsd for further examples.
"""
        print(msg)
        sys.exit()

    # Typing
    index: Union[Set[int], List[int], ndarray]

    # Set local view
    p_view: Optional[ndarray] = None
    q_view: Optional[ndarray] = None

    # Set local view
    if p_view is None:
        p_coord = copy.deepcopy(p_all)
        q_coord = copy.deepcopy(q_all)
        p_atoms = copy.deepcopy(p_all_atoms)
        q_atoms = copy.deepcopy(q_all_atoms)

    else:
        assert p_view is not None
        assert q_view is not None
        p_coord = copy.deepcopy(p_all[p_view])
        q_coord = copy.deepcopy(q_all[q_view])
        p_atoms = copy.deepcopy(p_all_atoms[p_view])
        q_atoms = copy.deepcopy(q_all_atoms[q_view])

    # Recenter to centroid
    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent

    rmsd_method: RmsdCallable
    reorder_method: Optional[ReorderCallable]

    # set rotation method
    if rotation_method == 'kabsch':
        rmsd_method = kabsch_rmsd
    elif rotation_method == 'quaternion':
        rmsd_method = quaternion_rmsd
    else:
        rmsd_method = rmsd

    # set reorder method
    reorder_method = None
    if reorder_method == 'hungarian':
        reorder_method = reorder_hungarian
    elif settings.reorder_method == 'brute':
        reorder_method = reorder_brute  # pragma: no cover

    # Save the resulting RMSD
    result_rmsd = None

    # Collect changes to be done on q coords
    q_swap = None
    q_reflection = None
    q_review = None

    if use_reflections:
        result_rmsd, q_swap, q_reflection, q_review = check_reflections(
            p_atoms,
            q_atoms,
            p_coord,
            q_coord,
            reorder_method=reorder_method,
            rmsd_method=rmsd_method,
        )

    elif use_ref_stereo:
        result_rmsd, q_swap, q_reflection, q_review = check_reflections(
            p_atoms,
            q_atoms,
            p_coord,
            q_coord,
            reorder_method=reorder_method,
            rmsd_method=rmsd_method,
            keep_stereo=True,
        )

    elif reorder:

        assert reorder_method is not None, \
                "Cannot reorder without selecting --reorder method"

        q_review = reorder_method(p_atoms, q_atoms, p_coord, q_coord)

    if q_review is not None:
        q_all_atoms = q_all_atoms[q_review]
        q_atoms = q_atoms[q_review]
        q_coord = q_coord[q_review]

        assert all(
            p_atoms == q_atoms
        ), "error: Structure not aligned. Please submit bug report at http://github.com/charnley/rmsd"

    # print result
    if output:

        if q_swap is not None:
            q_coord = q_coord[:, q_swap]

        if q_reflection is not None:
            q_coord = np.dot(q_coord, np.diag(q_reflection))

        q_coord -= centroid(q_coord)

        # Rotate q coordinates
        # TODO Should actually follow rotation method !Does this TODO matter?
        q_coord = kabsch_rotate(q_coord, p_coord)

        # center q on p's original coordinates
        q_coord += p_cent

        # done and done
        xyz = set_coordinates(q_all_atoms, q_coord, title=f"{settings.structure_b} - modified")
        print(xyz)

    else:

        if not result_rmsd:
            result_rmsd = rmsd_method(p_coord, q_coord)

        # Probably want to return this to loop over structures.
        print("{0}".format(result_rmsd))

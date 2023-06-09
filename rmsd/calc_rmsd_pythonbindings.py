import argparse, copy, gzip, re, sys
from pathlib import Path
from typing import Any, Iterator, List, Optional, Protocol, Set, Tuple, Union
from rmsd import * 
import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial import distance_matrix  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore

try:
    import qml  # type: ignore
except ImportError:  # pragma: no cover
    qml = None  # pragma: no cover

def get_rmsd(p_all_atoms, p_all, q_all_atoms, q_all, reorder = True,
             reorder_method = 'hungarian', rotation_method = 'kabsch',
             use_reflections = False, use_ref_stereo = False,
             print_rmsd = True) -> float:

    """

    p/q_all_atoms -> list of atomic numbers
    p/q_all -> np.ndarray of coordinates (N,3)

    reorder_method -> 'brute', 'hungarian'
    reorder_method -> 'kabsch', 'quaternion'

    """

    # Parse arguments
    # settings = parse_arguments(args)

    p_size = p_all.shape[0]
    q_size = q_all.shape[0]

    if not p_size == q_size:
        raise ValueError("error: Structures not same size")

    if np.count_nonzero(p_all_atoms != q_all_atoms) and not reorder:
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

    # rmsd_method: RmsdCallable
    # reorder_method: Optional[ReorderCallable]

    # set rotation method
    if rotation_method == 'kabsch':
        rmsd_method = kabsch_rmsd
    elif rotation_method == 'quaternion':
        rmsd_method = quaternion_rmsd
    else:
        rmsd_method = rmsd

    # set reorder method
    # reorder_method = None # this is an artifact of them
    # needing to pull from argparsed settings
    if reorder_method == 'hungarian':
        reorder_method = reorder_hungarian
    elif reorder_method == 'brute':
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


    # We don't really care about the mapped orientations, so this whole block
    # I am just commenting out.

    # if output:

        # if q_swap is not None:
            # q_coord = q_coord[:, q_swap]

        # if q_reflection is not None:
            # q_coord = np.dot(q_coord, np.diag(q_reflection))

        # q_coord -= centroid(q_coord)

        # # Rotate q coordinates
        # # TODO Should actually follow rotation method !Does this TODO matter?
        # q_coord = kabsch_rotate(q_coord, p_coord)

        # # center q on p's original coordinates
        # q_coord += p_cent

        # # done and done
        # xyz = set_coordinates(q_all_atoms, q_coord, title=f"{structure_b} - modified")
        # print(xyz)

    if not result_rmsd:
        result_rmsd = rmsd_method(p_coord, q_coord)

    # Probably want to return this to loop over structures.
    # print("{0}".format(result_rmsd))
    if print_rmsd:
        print(result_rmsd)

    return result_rmsd

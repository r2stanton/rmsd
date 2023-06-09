import ase.io
# from calc_rmsd_pythonbindings import get_rmsd
from rmsd import *
from context import RESOURCE_PATH

def test_shuffled_atom_ordering():
    base = ase.io.read(RESOURCE_PATH / "base.xyz")
    shuf = ase.io.read(RESOURCE_PATH / "shuffled_atom_order.xyz")
    
    r = get_rmsd(base.get_atomic_numbers(), base.positions,
                 shuf.get_atomic_numbers(), shuf.positions,
                 rotation_method = "kabsch",
                 reorder_method = "hungarian",
                 reorder = True,
                 print_rmsd = False)

    # Obtain CLI result with:
    # calculate_rmsd --reorder --reorder-method hungarian base.xyz shuffled_atom_order.xyz
    assert abs(r - 6.792478067956709e-16) < 1e-6 , \
            "Python bindings don't match CLI"

def test_shuff_rot_trans():
    base = ase.io.read(RESOURCE_PATH / "base.xyz")
    shuf = ase.io.read(RESOURCE_PATH / "shuffled_then_rot_trans.xyz")
    
    r = get_rmsd(base.get_atomic_numbers(), base.positions,
                 shuf.get_atomic_numbers(), shuf.positions,
                 rotation_method = "kabsch",
                 reorder_method = "hungarian",
                 reorder = True,
                 print_rmsd = False)

    assert r < 1e-6 , \
            "Doesn't detect identical molecule shuffled, rotated, and translated"


def test_rattle():
    base = ase.io.read(RESOURCE_PATH / "base.xyz")
    ratt = ase.io.read(RESOURCE_PATH / "rattled.xyz")
    
    r = get_rmsd(base.get_atomic_numbers(), base.positions,
                 ratt.get_atomic_numbers(), ratt.positions,
                 rotation_method = "kabsch",
                 reorder_method = "hungarian",
                 reorder = True,
                 print_rmsd = False)

    # Obtain CLI result with:
    # calculate_rmsd --reorder --reorder-method hungarian base.xyz rattled.xyz
    assert abs(r - 0.14950270597012547) < 1e-6 , \
            "Python bindings don't match CLI"

def test_rattle_shuffle():
    # Then rattle + shuffle
    base = ase.io.read(RESOURCE_PATH / "base.xyz")
    ratt_s = ase.io.read(RESOURCE_PATH / "rattled_shuffled.xyz")
    
    r = get_rmsd(base.get_atomic_numbers(), base.positions,
                 ratt_s.get_atomic_numbers(), ratt_s.positions,
                 rotation_method = "kabsch",
                 reorder_method = "hungarian",
                 reorder = True,
                 print_rmsd = False)

    # Obtain CLI result with:
    # calculate_rmsd --reorder --reorder-method hungarian base.xyz rattled.xyz
    assert abs(r - 0.14950270597012547) < 1e-6 , \
            "Python bindings don't match CLI"

def test_base_package():
    # This tests that the RMSD is ~0 for a structure which I manually rotated
    # and translated from the original base.xyz structure.

    base = ase.io.read(RESOURCE_PATH / "base.xyz")
    r_t = ase.io.read(RESOURCE_PATH / "manual_rotate_and_translate.xyz")
    
    r = get_rmsd(base.get_atomic_numbers(), base.positions,
                 r_t.get_atomic_numbers(), r_t.positions,
                 rotation_method = "kabsch",
                 reorder_method = "hungarian",
                 reorder = True,
                 print_rmsd = False)

    assert r < 1e-6 , \
            "Package doesn't correctly detect identical molecule"



    

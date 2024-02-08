"""
Unit and regression test for the molecool package.
"""

# Import package, test suite, and other packages as needed
import sys
import numpy as np

import pytest

import molecool

@pytest.fixture
def methane_molecule():
    coordinates = np.array([[1,1,1],[2.4,1,1],[-0.4,1,1],[1,1,2.4],[1,1,-0.4]])
    symbols = ['C', 'H', 'H', 'H', 'H']
    return symbols, coordinates

def test_molecool_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "molecool" in sys.modules

def test_calculate_distance():
    """Make sure calc distance works"""
    r1 = np.array([0,0,0])
    r2 = np.array([0,1,0])
    calculated_distance = molecool.calculate_distance(r1, r2)

    assert calculated_distance == 1

def test_calculate_angle():
    """Is the angle calculation correct?"""
    r1 = np.array([0,0,-1])
    r2 = np.array([0,0,0])
    r3 = np.array([1,0,0])

    # v1 = r1 - r2
    # v2 = r3 - r2

    # my_angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    my_angle = np.pi / 2

    your_angle = molecool.calculate_angle(r1, r2, r3, degrees=False)

    assert my_angle == your_angle

def test_molecular_mass(methane_molecule):
    """Make sure molecular masses can be computed correctly"""
    # symbols = ['C', 'H', 'H', 'H', 'H']
    symbols = methane_molecule[0]
    calculated_mass = molecool.calculate_molecular_mass(symbols)
    actual_mass = 16.04
    assert pytest.approx(actual_mass, abs=1e-2) == calculated_mass

def test_build_bond_list(methane_molecule):
    coordinates = methane_molecule[1]
    bonds = molecool.build_bond_list(coordinates)
    assert len(bonds) == 4


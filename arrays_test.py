import pytest
import stride

def test_onedim_arrays():
    a = stride.array([1, 2, 3])

    assert a.shape == (3,)
    assert a[0] == 1 and a[1] == 2 and a[2] == 3

def test_multidim_arrays():
    a = stride.array([[1], [2], [3]])
    assert a.shape == (3, 1)

    a = stride.array([[1, 2, 3]])
    assert a.shape == (1, 3)

    a = stride.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert a.shape == (3, 3)
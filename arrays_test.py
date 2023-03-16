import pytest
import stride


def test_zeros():
    a = stride.zero_array((3, ))
    assert a.shape == (3,)
    assert a.payload == [0, 0, 0]

    a = stride.zero_array((3, 4))
    assert a.shape == (3, 4)
    assert a.payload == [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

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

def test_broadcast_failure():
    a = stride.array([1, 2, 3])
    b = stride.broadcast_to(a, (2, 3))
    assert b.payload == [[1, 2, 3], [1, 2, 3]]
    
def test_out():
    a = stride.array([1, 2, 3])
    b = stride.out(a, 2)
    print("fek")
    assert b == 0
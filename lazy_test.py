import pytest
from lazy import lazy_range
from playground import decompose

def _test_access():
    a = lazy_range(1, 26, (5, 5))

    for i in range(0, 25):
        ii = decompose(i, (5, 5))
        print("!!", ii, i+1)
        assert a[ii] == i+1

def test_access_reshape():
    a = lazy_range(1, 7, (2, 3))
    for i in range(0, 2):
        for j in range(0, 3):
            assert a[i, j] == i*3 + j + 1
    b = a.reshape((3, 2))
    print("!!!!", b)
    for i in range(0, 3):
        for j in range(0, 2):
            assert b[i, j] == i*2 + j + 1
    c = b.reshape((6,))
    for i in range(0, 6):
        assert c[i] == i + 1

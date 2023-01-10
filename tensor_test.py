import pytest
import stride

def test_onedim_getandsetitem():
    t = stride.zeros(5)

    assert t[0] == 0
    for i in range(5):
        t[i] = i * 2

    for i in range(5):
        assert t[i] == i * 2

def test_twodim_getandsetitem():
    t = stride.zeros(5, 4)

    assert t[0, 2] == 0
    for i in range(5):
        for j in range(4):
            t[i, j] = i + j + 2

    for i in range(5):
        for j in range(4):
            assert t[i, j] == i + j + 2
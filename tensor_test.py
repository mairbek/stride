import pytest
import stride


def test_normalize_index():
    t = stride.zeros(5)
    assert t.normalize_index(0) == (slice(0, 1), slice(0, 1))

    t = stride.zeros(5, 4)
    assert t.normalize_index(1) == (slice(1, 2), slice(0, 4))
    assert t.normalize_index((1, 2)) == (slice(1, 2), slice(2, 3))
    assert t.normalize_index((1, slice(0, 3))) == (slice(1, 2), slice(0, 3))
    assert t.normalize_index((slice(0, 3), 2)) == (slice(0, 3), slice(2, 3))


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


def test_onedim_range():
    t = stride.zeros(5)
    for i in range(5):
        t[i] = i * 2

    assert t[1:3] == [2, 4]


def test_twodim_range():
    t = stride.zeros(5, 4)
    for i in range(5):
        for j in range(4):
            t[i, j] = i + j + 2

    assert t[1, 1:3] == [2, 4]

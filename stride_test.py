import pytest

import stride as st


def test_zeros():
    a = st.zeros((10, ))
    assert a.shape == (10, )
    assert all(i == 0 for i in a)

    a = st.zeros((10, 10))
    assert a.shape == (10, 10)


def test_array():
    a = st.array([[1, 2, 3], [4, 5, 6]])
    assert a.shape == (2, 3)
    assert a[0] == [1, 2, 3]
    assert a[1] == [4, 5, 6]

def test_complex():
    # TODO fix
    a = st.arange(1, 26).reshape(5, 5)
    b = a[1:4, 1:4]
    assert b.shape == (3, 3)
    c = b.reshape((9,))
    assert c == [7, 8, 9, 12, 13, 14, 17, 18, 19]
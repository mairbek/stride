import pytest
import stride as st


def test_access_reshape():
    a = st.arange(1, 7).reshape(2, 3)
    for i in range(0, 2):
        for j in range(0, 3):
            assert a[i, j] == i*3 + j + 1
    b = a.reshape((3, 2))
    for i in range(0, 3):
        for j in range(0, 2):
            assert b[i, j] == i*2 + j + 1
    c = b.reshape((6,))
    for i in range(0, 6):
        assert c[i] == i + 1


def test_slice_1d():
    a = st.arange(1, 7)
    b = a.subrange(((1, 1, 4),))
    for i in range(0, 3):
        assert b[i] == i + 2

    c = a.subrange(((0, 2, 7),))
    for i in range(0, 3):
        assert c[i] == i*2 + 1


def test_slice_in_slice():
    a = st.arange(1, 26)
    b = a.subrange(((1, 1, 8),))
    c = b.subrange(((2, 1, 5),))

    for i in range(0, 3):
        assert c[i] == i + 4

    d = a.subrange(((2, 2, 12),))
    dd = as_1d_list(d)
    assert dd == [3, 5, 7, 9, 11]

    e = d.subrange(((0, 2, 4),))
    ee = as_1d_list(e)
    assert ee == [3, 7]


def as_1d_list(larr):
    result = []
    for i in range(larr.shape[0]):
        result.append(larr[i])
    return result


def as_2d_list(larr):
    result = []
    for i in range(larr.shape[0]):
        result.append([])
        for j in range(larr.shape[1]):
            result[i].append(larr[i, j])
    return result


def as_3d_list(larr):
    result = []
    for i in range(larr.shape[0]):
        result.append([])
        for j in range(larr.shape[1]):
            result[i].append([])
            for k in range(larr.shape[2]):
                result[i][j].append(larr[i, j, k])
    return result


def test_slice_in_2d():
    a = st.arange(0, 25).reshape(5, 5)
    b = a.subrange(((1, 1, 5), (1, 1, 5)))
    c = b.subrange(((1, 1, 4), (1, 1, 4)))
    d = a.subrange(((3, 1, 5), (2, 1, 4)))
    e = a.subrange(((0, 2, 5), (0, 1, 5)))
    f = a.subrange(((0, 2, 5), (0, 3, 5)))
    g = a.subrange(((1, 2, 5), (0, 1, 5)))
    h = g.subrange(((0, 1, 2), (1, 2, 5)))

    assert as_2d_list(a) == [[0, 1, 2, 3, 4],
                             [5, 6, 7, 8, 9],
                             [10, 11, 12, 13, 14],
                             [15, 16, 17, 18, 19],
                             [20, 21, 22, 23, 24]]

    assert as_2d_list(b) == [[6, 7, 8, 9],
                             [11, 12, 13, 14],
                             [16, 17, 18, 19],
                             [21, 22, 23, 24]]

    assert as_2d_list(c) == [[12, 13, 14],
                             [17, 18, 19],
                             [22, 23, 24]]

    assert as_2d_list(d) == [[17, 18], [22, 23]]
    assert as_2d_list(e) == [[0, 1, 2, 3, 4],
                             [10, 11, 12, 13, 14],
                             [20, 21, 22, 23, 24]]

    assert as_2d_list(f) == [[0, 3],
                             [10, 13],
                             [20, 23]]
    assert as_2d_list(g) == [[5, 6, 7, 8, 9],
                             [15, 16, 17, 18, 19]]
    assert as_2d_list(h) == [[6, 8],
                             [16, 18]]


def test_slice_in_3d():
    a = st.arange(1, 13).reshape(2, 2, 3)
    b = a.subrange(((0, 2, 2), (0, 1, 2), (0, 1, 3)))
    c = a.subrange(((0, 1, 2), (0, 2, 2), (0, 1, 3)))
    bb = as_3d_list(b)
    assert bb == [[[1, 2, 3], [4, 5, 6]]]
    cc = as_3d_list(c)
    assert cc == [[[1, 2, 3]], [[7, 8, 9]]]


def test_slicing():
    a = st.arange(1, 26).reshape(5, 5)
    b = a[0:3, 0:3]
    bb = as_2d_list(b)
    assert b.shape == (3, 3)
    assert bb == [[1, 2, 3], [6, 7, 8], [11, 12, 13]]

    c = a[1:4, 1:4]
    cc = as_2d_list(c)
    assert c.shape == (3, 3)
    assert cc == [[7, 8, 9], [12, 13, 14], [17, 18, 19]]

    d = a[0:5:3, 1:5:2]
    assert d.shape == (2, 2)
    dd = as_2d_list(d)
    assert dd == [[2, 4], [17, 19]]


def test_unwrap():
    a = st.arange(1, 26).reshape(5, 5)
    b = a[1]
    assert b.shape == (5,)
    bb = as_1d_list(b)
    assert bb == [6, 7, 8, 9, 10]
    c = a[:, 1]
    assert c.shape == (5,)
    cc = as_1d_list(a[:, 1])
    assert cc == [2, 7, 12, 17, 22]
    d = a[1, 1]
    e = a[1][1]
    assert d == 7
    assert d == e


def test_ndindex():
    a = st.arange(1, 5).reshape(2, 2)
    b = list(a.ndindex())
    assert b == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_ndenumerate():
    a = st.arange(1, 5).reshape(2, 2)
    b = list(a.ndenumerate())
    print(b)
    assert b == [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)]


def test_1d_iter():
    a = st.arange(1, 5)
    for i, j in zip(a, range(1, 5)):
        assert i == j


def test_2d_iter():
    a = st.arange(1, 10).reshape(3, 3)
    for i, j in zip(a, range(0, 4)):
        assert i.shape == (3, )
        assert as_1d_list(i) == [1 + j*3 + k for k in range(0, 3)]


def test_1d_equals():
    a = st.arange(1, 5)
    b = [1, 2, 3, 4]
    assert a == b
    c = [1, 2, 3, 5]
    assert a != c
    d = st.arange(1, 5)
    assert a == d


def test_2d_equals():
    a = st.arange(1, 10).reshape(3, 3)
    b = st.arange(1, 10).reshape(3, 3)
    assert a == b
    c = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert a == c
    d = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert a != d


def test_to_list():
    a = st.arange(1, 4)
    assert a.to_list() == [1, 2, 3]
    a = st.arange(1, 10).reshape((3, 3))
    assert a.to_list() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert a[1:, 1:].to_list() == [[5, 6], [8, 9]]
    a = st.arange(1, 13).reshape(2, 2, 3)
    assert a.to_list() == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]


def test_repr():
    a = st.arange(1, 4)
    assert repr(a) == "array([1, 2, 3])"
    a = st.arange(1, 10).reshape((3, 3))
    assert repr(a) == "array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])"
    a = st.arange(1, 13).reshape(2, 2, 3)
    assert repr(a) == "array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])"

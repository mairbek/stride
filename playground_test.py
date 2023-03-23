import pytest
from playground import as_strided, decompose, reshape, flatten


def test_reshape_precond():
    a = reshape(list(range(1, 26)), (5, 5))
    assert flatten(a) == list(range(1, 26))


def test_as_strided_2d_to1d():
    a = reshape(list(range(1, 26)), (5, 5))
    assert as_strided(a, (1, ), (3, )) == [1, 2, 3]
    assert as_strided(a, (1, ), (8, )) == [1, 2, 3, 4, 5, 6, 7, 8]

    # flatten 2d array
    assert as_strided(a, (1, ), (25, )) == list(range(1, 26))
    # skip every other element
    assert as_strided(a, (2, ), (3, )) == [1, 3, 5]
    # slice first column
    assert as_strided(a, (5, ), (4, )) == [1, 6, 11, 16]
    # slice a diagonal
    assert as_strided(a, (6, ), (5, )) == [1, 7, 13, 19, 25]
    # repeat the first element
    assert as_strided(a, (0, ), (5, )) == [1, 1, 1, 1, 1]


def test_as_strided_2d_to_2d():
    a = reshape(list(range(1, 26)), (5, 5))

    # simple 2d slicing
    assert as_strided(a, (5, 1), (3, 4)) == [[1,  2,  3,  4],
                                             [6,  7,  8,  9],
                                             [11, 12, 13, 14]]

    # slice a zig-zag
    assert as_strided(a, (6, 1), (4, 2)) == [[1,  2],
                                             [7,  8],
                                             [13, 14],
                                             [19, 20]]

    # sparse slicing
    assert as_strided(a, (10, 2), (3, 3)) == [[1,  3,  5],
                                              [11, 13, 15],
                                              [21, 23, 25]]
    # transpose 2d array
    assert as_strided(a, (1, 5), (3, 3)) == [[1, 6, 11],
                                             [2, 7, 12],
                                             [3, 8, 13]]
    # repeat first column 4 times
    assert as_strided(a, (5, 0), (5, 4)) == [[1,  1,  1,  1],
                                             [6,  6,  6,  6],
                                             [11, 11, 11, 11],
                                             [16, 16, 16, 16],
                                             [21, 21, 21, 21]]


def test_as_strided_1d_to_2d():
    a = list(range(1, 13))

    # Reshape 1D array to 2D array
    assert as_strided(a, (4, 1), (3, 3)) == [[1,  2,  3],
                                             [5,  6,  7],
                                             [9, 10, 11]]
    # Slide a 1d window
    assert as_strided(a, (1, 1), (8, 3)) == [[1,  2,  3],
                                             [2,  3,  4],
                                             [3,  4,  5],
                                             [4,  5,  6],
                                             [5,  6,  7],
                                             [6,  7,  8],
                                             [7,  8,  9],
                                             [8,  9, 10]]


def test_as_strided_3d_to_2d():
    a = reshape(list(range(1, 13)), (3, 2, 2))
    assert as_strided(a, (4, 1), (3, 4)) == [[1,  2,  3,  4],
                                             [5,  6,  7,  8],
                                             [9, 10, 11, 12]]


def test_as_strided_1d_to_3d():
    # Reshape 1D array to 3D array
    a = list(range(1, 13))
    assert as_strided(a, (6, 3, 1), (2, 2, 3)) == [[[1,  2,  3],
                                                    [4,  5,  6]],
                                                   [[7,  8,  9],
                                                    [10, 11, 12]]]


def test_reshape():
    flat = list(range(0, 4))
    a2d = reshape(flat, (2, 2))
    assert a2d == [[0, 1], [2, 3]]
    reflat = flatten(a2d)
    assert reflat == flat


def test_decompose_2d():
    n = 2
    m = 3
    for i in range(n):
        for j in range(m):
            nn = i * m + j
            print(nn, (i, j))
            assert decompose(nn, (n, m)) == (i, j)

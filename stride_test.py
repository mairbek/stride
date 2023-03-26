import stride as st

def test_zeros():
    a = st.zeros((10, ))
    assert a.shape == (10, )
    assert all(i == 0 for i in a)

    a = st.zeros((10, 10))
    assert a.shape == (10, 10)

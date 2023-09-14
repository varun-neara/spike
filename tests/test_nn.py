import numpy as np

from spike import nn



def test_dense_init():
	layer = nn.Dense.xavier_init(30, 10)
	data = np.random.random((100, 30))
	out = layer.forward(data)
	assert list(out.shape) == [100, 10]

def test_activation():
	activation = nn.Relu()
	data = np.random.random((100, 30))
	out = activation.forward(data)
	assert list(out.shape) == (100, 30)
	assert np.min(out) >= 0
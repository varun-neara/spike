import numpy as np
import pytest

from spike import nn


@pytest.fixture
def data():
	return np.random.random((100, 30))



def test_dense_init(data):
	layer = nn.Dense.xavier_init(30, 10)
	out = layer.forward(data)
	assert list(out.shape) == [100, 10]

@pytest.mark.parametrize("module", [nn.Relu, nn.Softmax, lambda : nn.Dense.xavier_init(30, 10)])
def test_mods(data, module):
	layer = module()
	out = layer.forward(data)

def test_module(data):
	mod = nn.Module([nn.Dense.xavier_init(30, 10), nn.Dense.xavier_init(10, 3)])
	out = mod.forward(data)


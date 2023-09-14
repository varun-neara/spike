import numpy as np
import attr


class Layer:
	def forward(self, *args, **kwargs):
		raise NotImplementedError()

@attr.define
class Dense:
	weight: np.array
	bias: np.array

	@classmethod
	def xavier_init(cls, num_inputs: int, num_outputs: int):
		sd = np.sqrt(2/(num_outputs + num_inputs))
		weight = np.random.normal(loc=0, scale=sd, size=(num_inputs, num_outputs))
		bias = np.zeros(num_outputs)
		return cls(weight, bias)

	def forward(self, x: np.array):
		"""
		Args:
			inputs: np.array of shape [N, feat_in]
		"""
		assert len(x.shape) == 2
		assert x.shape[1] == self.weight.shape[0]

		return x @ self.weight  + self.bias


@attr.define
class Relu:
	def forward(self, x):
		return np.max(x, 0)


@attr.define
class Softmax:
	def forward(self, x):
		exp = np.exp(x)
		return exp/np.sum(exp)


@attr.define
class Module:
	layers: list[Layer] = attr.field(factory=list)

	def forward(self, x):
		for l in self.layers:
			x = l.forward(x)
		return x


def main():
	model = Module([Dense(50, 25), Relu(), Dense(25, 10), Relu(), Dense(10, 3), Softmax()])
	return model
	



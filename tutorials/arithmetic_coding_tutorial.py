from core.prob_dist import ProbabilityDist
from core.data_block import DataBlock

# define a sample distribution
prob = ProbabilityDist({"A": 0.2, "B": 0.4, "C": 0.4})

# define a sample input
data = DataBlock(["B", "A", "C", "B"])

# initalize low, high values
low, high = 0.0, 1.0

# recursively shrink the range
print(f"initial range: low {low:.4f}, high: {high:.4f}")
for i, s in enumerate(data.data_list):
    # define some intermediate variables
    rng = high - low
    low = low + prob.cumulative_prob_dict[s] * rng
    high = low + prob.probability(s) * rng
    print(f"{i}: symbol {s}, low {low:.4f}, high: {high:.4f}")

from utils.bitarray_utils import float_to_bitarrays

mid = (low + high) / 2
print(mid)
_, float_bitarray = float_to_bitarrays(mid, max_precision=20)
print(float_bitarray)

import numpy as np

k = np.ceil(-np.log2(high - low) + 1)

_, float_bitarray = float_to_bitarrays(mid, max_precision=int(k))
print(float_bitarray)

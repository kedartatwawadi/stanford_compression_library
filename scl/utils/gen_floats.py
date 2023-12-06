import numpy as np
import os

# half-prec floats, double-prec floats, and long doubles
list_of_float_types = [np.float16, np.float32, np.float64]
outdir = "./float_data/"

type_map = {
    np.float16: "half_prec",
    np.float32: "single_prec",
    np.float64: "double_prec",
}

def gen_floats(float_type, negative_proportion: float = 0.1, num_floats: int = int(1e2)):
    assert negative_proportion >= 0 and negative_proportion <= 1, "negative_proportion must be between 0 and 1"

    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123)))
    values = rs.rand(num_floats)
    signs = rs.choice([-1, 1], p=[negative_proportion, 1-negative_proportion], size=num_floats)
    values = np.multiply(values, signs)

    arr = np.array(values, dtype=float_type)
    return arr



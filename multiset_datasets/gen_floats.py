import numpy as np
import os

num_floats = int(1e2)

# half-prec floats, double-prec floats, and long doubles
list_of_float_types = [np.float16, np.float32, np.float64, np.float128] 
outdir = "./float_data/"

type_map = {
    np.float16: "half_prec",
    np.float32: "single_prec",
    np.float64: "double_prec",
    np.float128: "long_double",
}

if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

def gen_floats(float_types, negative_proportion, fname_base):
    assert negative_proportion >= 0 and negative_proportion <= 1, "negative_proportion must be between 0 and 1"

    for t in float_types:
        rs.set_state(start_state)
        values = rs.rand(num_floats).astype(np.float128)
        signs = rs.choice([-1, 1], p=[negative_proportion, 1-negative_proportion], size=num_floats)
        values = np.multiply(values, signs)

        arr = np.array(values, dtype=t)
        arr.tofile(f"{outdir}/{fname_base}_negs_{negative_proportion}_{type_map[t]}.bin", sep='\n')

negative_proportion = 0.1
rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123)))
start_state = rs.get_state()

gen_floats(list_of_float_types, negative_proportion, "float_vals")



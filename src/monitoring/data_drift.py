import numpy as np
import pandas as pd

def population_stability_index(ref: pd.Series, new: pd.Series) -> float:
    ref_dist = ref.value_counts(normalize=True)
    new_dist = new.value_counts(normalize=True)

    all_cats = set(ref_dist.index).union(new_dist.index)
    psi = 0.0

    for c in all_cats:
        p = ref_dist.get(c, 1e-6)
        q = new_dist.get(c, 1e-6)
        psi += (p - q) * np.log(p / q)

    return float(psi)

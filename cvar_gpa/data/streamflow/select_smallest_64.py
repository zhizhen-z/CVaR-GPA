"""Build the streamflow dataset of the paper: the 64 gauges with the smallest drainage area among the
gap-free gauges of ohio_all_complete_dv_2005_2024.npz (sort by drain_area_km2, take the first 64).
Output: ohio_smallest64_dv_2005_2024.npz, same keys."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "ohio_all_complete_dv_2005_2024.npz"), allow_pickle=True)
idx = np.argsort(z["drain_area_km2"], kind="stable")[:64]
out = {k: (z[k][:, idx] if k == "discharge" else z[k] if k == "time_days" else z[k][idx]) for k in z.keys()}
np.savez(os.path.join(HERE, "ohio_smallest64_dv_2005_2024.npz"), **out)
print("wrote ohio_smallest64_dv_2005_2024.npz:", out["discharge"].shape, "areas %.1f-%.1f km2" % (out["drain_area_km2"].min(), out["drain_area_km2"].max()))

from pathlib import Path
import h5py
from hiutils.visutils import render_gaussian, write_img
import numpy as np

ours_worst_dir = Path("/home/blackhole/dreamy1534/Projects/iccv2023-spaghetti/part_ldm/figure_scripts/output_data/collect_our_sdedit_results/shapeformer_worst_case_ours")
gmm_path = ours_worst_dir / "gmms.hdf5"

with h5py.File(gmm_path) as f:
    all_src_gmms = np.array(f['all_src_gmms'])
    # all_src_gmms_ablated = f['all_src_gmms_ablated']
    # all_src_sids = f['all_src_sids']
    # all_src_pids = f['all_src_pids']


# for i in track(range(len(all_src_gmms))):
    # ingmm = render_gaussian(all_src_gmms[i])
    # ingmm_wo_part = render_gaussian(all_)

ingmm = render_gaussian(all_src_gmms[0], camType='orthographic')
write_img(ingmm, "./gausvis.png")

import pyvista as pv
from pyvista import examples

bunny = examples.download_bunny()

###############################################################################
# No Anti-Aliasing
# ~~~~~~~~~~~~~~~~
# First, let's show a plot without any anti-aliasing.

# obtained with `cpos = pl.show(return_cpos=True)`
cpos = [(-0.08566, 0.18735, 0.20116), (-0.05332, 0.12168, -0.01215), (-0.00151, 0.95566, -0.29446)]

pl = pv.Plotter()
pl.add_mesh(bunny, show_edges=True)
pl.disable_anti_aliasing()
pl.camera_position = cpos
pl.show()


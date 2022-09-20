from src.plot import Plot
from src.mesh import Mesh
from src.model import FEM_Model
import matplotlib.pyplot as plt

# --- Maillage ---
mesh = Mesh(2, debug=False)
mesh.add_node([0, 0])
mesh.add_node([3, 0])
mesh.add_element([1, 2], "entrait", "r", 0.15, 0.15, 10)

# --- Modèle ---
f = FEM_Model(mesh)
f.apply_load([0, -1000, 0], 11)
f.apply_bc([1, 1, 1], 1)
f.solver_frame()

f.U_table()
f.R_table()
print("stress :", f.stress)

res = f.get_res()

# --- POST-PROCESSING ---
print("Post-processing...")
# f.plot_disp_f_ex(scale=1e2)
post = Plot(res,mesh, f)
post._plot_elements_2d()
post._plot_loads_2d()
post.plot_diagram()
#post.plot_disp(scale=1e0, r=150, dir='y', pic=False, path="../test/")

"""
post.plot_stress(s='sx')
post.plot_stress(s='sf')
post.plot_stress(s='ty')
post.plot_stress(s='svm')
"""

plt.show()
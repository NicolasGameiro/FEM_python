### Cas 1 : poutre encastrée-libre effort ponctuel à son extrémité

# Packages
import matplotlib.pyplot as plt
import numpy as np
from src.mesh import Mesh
from src.model import FEM_Model

### Analytique
F = -1000 #N
L = 10 #m
E = 210E9 #Pa
h, b = 0.15, 0.15
I = h * b ** 3 / 12 #m4
x = np.linspace(0, L, 20)
d = lambda x : F * x ** 2 * (3*L-x) / (6*E*I)
w = d(x)

#1 ----- Mesh creation -----

mesh = Mesh(2, debug=False)

nb_element = 10

# Create nodes
for i in range(nb_element+1):
    mesh.add_node([i, 0])

# Create elements
for i in range(nb_element):
    mesh.add_element([i+1, i+2], "entrait", "r", h, b)

# Plot mesh
mesh.plot_mesh()
plt.show()

#2 ----- Model creation -----

f = FEM_Model(mesh)

# Apply force
f.apply_load([0, F, 0], nb_element + 1)

# Apply boundary conditions
f.apply_bc([1, 1, 1], 1)

# plot model
f.plot_model()

# Solve KU = F
f.solver_frame()

# plot diagram
f.plot_diagram()

# Print results
f.U_table()
f.R_table()

#3 ----- POST-PROCESSING -----
disp_x = f.get_displacement()
disp_y = f.get_displacement(dir="Y")

#4 ----- ANALYTIC VS FEM -----
print('f_max (analytique) = ', np.format_float_scientific(max(abs(w)), precision=2, exp_digits=2))
print('f_max (FEM) = ', np.format_float_scientific(max(disp_y, key=abs), precision=2, exp_digits=2))
err = abs(max(abs(w)) - max(disp_y, key=abs))
print("erreur max :", err[0])




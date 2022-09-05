import matplotlib.pyplot as plt

from mesh import Mesh
from model import FEM_Model
from plot import Plot
from log import logger
from __version__ import __version__
from __init__ import MODULE_NAME


def test_3d():
    m1 = Mesh(dim=3)
    p = 6.5
    h = 2
    h_mur = 2.5
    L = 6
    m1.add_node([0, 0, 0])
    m1.add_node([0, 0, h_mur])
    m1.add_node([0, p / 2, h_mur + h])
    m1.add_node([0, p, h_mur])
    m1.add_node([0, p, 0])
    m1.add_node([L, p / 2, h_mur + h])
    m1.add_element([1, 2], "poteau", "k", 15, 15)
    m1.add_element([2, 3], "arba", "r", 22, 12)
    m1.add_element([3, 4], "arba", "r", 22, 12)
    m1.add_element([4, 2], "entrait", "b", 22, 10)
    m1.add_element([4, 5], "poteau", "k", 15, 15)
    m1.add_element([3, 6], "panne faitiere", 'g', 22, 12)
    # m1.geom()
    m1.node_table()
    # plt.show()
    f = FEM_Model(m1)
    f.apply_distributed_load(1, [3, 6])
    f.plot_forces3D(type='nodal')
    f.apply_bc([1, 1, 1, 0, 0, 0], 1)
    f.apply_bc([1, 1, 1, 1, 1, 1], 5)
    f.apply_bc([1, 1, 1, 1, 1, 1], 6)
    f.solver_frame()
    res = f.get_res()
    # ----- POST-PROCESSING -----
    logger.info("Post-processing...")
    # f.plot_disp_f_ex(scale=1e2)
    post = Plot(res, mesh)
    post.plot_disp_f_3D(dir="x")
    plt.show()
    f.U_table()
    f.R_table()
    return


def validation_3d():
    m1 = Mesh(dim=3)
    m1.add_node([0, 0, 0])
    m1.add_node([0, 0, 1])
    m1.add_node([0, 1, 1])
    m1.add_node([1, 1, 1])
    m1.add_element([1, 2], "poteau", "k", 10, 10)
    m1.add_element([2, 3], "arba", "r", 10, 10)
    m1.add_element([3, 4], "arba", "r", 10, 10)
    m1.node_table()
    m1.plot_mesh()
    f = FEM_Model(m1)
    f.apply_distributed_load(1, [2, 3])
    f.apply_load([0, 0, -0.1, 0, 0, 0], 4)
    f.apply_bc([1, 1, 1, 1, 1, 1], 1)
    # f.apply_bc([1, 1, 1, 1, 1, 1], 4)
    f.solver_frame()
    f.U_table()
    f.R_table()
    #f.S_table()
    res = f.get_res()
    # ----- POST-PROCESSING -----
    logger.info("Post-processing...")
    # f.plot_disp_f_ex(scale=1e2)
    post = Plot(res, m1, f)
    post.plot_forces3D(type = 'dist')
    plt.show()
    return


def validation_2d():
    mesh = Mesh(2, [], [], debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([0, 10])  # inches
    mesh.add_node([10, 10])  # inches
    mesh.add_element([1, 2], "barre", "b", 15, 15, 5)
    mesh.add_element([2, 3], "barre", "b", 15, 15, 5)
    # mesh.geom()

    f = FEM_Model(mesh)
    f.apply_distributed_load(10, [6, 11])
    f.apply_bc([1, 1, 1], 1)
    # f.apply_bc([0,1,0],3)
    # print(f.get_bc())
    f.plot_forces(type='dist', pic=False)
    f.solver_frame()
    # f.plot_disp_f(dir='x', pic = True)
    # f.plot_disp_f(dir='y' , pic = True)
    f.plot_disp_f(dir='sum', pic=True)
    # f.plot_disp_f_ex()
    S = f.stress()
    f.plot_stress(scale=1e2, r=100, s='sf', pic=False, path="./")
    f.S_table()
    f.U_table()
    f.R_table()


def test_2d():
    logger.info(f"===== {MODULE_NAME} {__version__} =====")

    # ----- MESHING -----
    mesh = Mesh(2, [], [], debug=True)
    p = 6.5
    h = 2.5
    mesh.add_node([0, 0])
    mesh.add_node([p / 2, 0])
    mesh.add_node([p, 0])
    mesh.add_node([p / 2, h])
    mesh.add_node([p / 4, h / 2])
    mesh.add_node([3 * p / 4, h / 2])
    mesh.add_element([1, 2], "entrait", "r", 22, 10)
    mesh.add_element([2, 3], "entrait", "r", 22, 10)
    mesh.add_element([3, 6], "arba", "g", 20, 8)
    mesh.add_element([6, 4], "arba", "g", 20, 8)
    mesh.add_element([4, 5], "arba", "g", 20, 8)
    mesh.add_element([5, 1], "arba", "g", 20, 8)
    mesh.add_element([4, 2], "poinçon", "b", 10, 10)
    mesh.add_element([2, 5], "jdf", "m", 10, 10)
    mesh.add_element([2, 6], "jdf", "m", 10, 10)
    mesh.plot_mesh()

    # ----- ASSEMBLING SYSTEM MATRICES -----
    logger.info("Assembling matrices...")

    # ----- SOLVING -----
    logger.info("Solving...")
    f = FEM_Model(mesh)
    # f.apply_load([0,-1000,0],4)
    f.apply_bc([1, 1, 1], 1)
    f.apply_bc([1, 1, 1], 3)
    f.apply_distributed_load(2000, [1, 4])
    f.apply_distributed_load(2000, [4, 3])
    f.solver_frame()
    res = f.get_res()
    # ----- POST-PROCESSING -----
    logger.info("Post-processing...")
    post = Plot(res,mesh)
    post.plot_mesh_2D()
    post.plot_stress(s='sx')
    post.plot_stress(s='sf')
    post.plot_stress(s='ty')
    post.plot_stress(s='svm')
    post.plot_forces()
    plt.show()


def test_cantilever():
    mesh = Mesh(2, debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([3, 0])
    mesh.add_element([1, 2], "entrait", "r", 0.15, 0.15, 10)
    print(mesh.get_node_lims())
    f = FEM_Model(mesh)
    f.apply_load([0, -1000, 0], 11)
    print("load" , f.load)
    f.apply_bc([1, 1, 1], 1)
    f.solver_frame()
    f.U_table()
    f.R_table()
    print("stress :", f.S)
    #f.S_table()
    res = f.get_res()
    print("forces : ", res )
    # ----- POST-PROCESSING -----
    logger.info("Post-processing...")
    # f.plot_disp_f_ex(scale=1e2)
    post = Plot(res,mesh, f)
    #post.plot_mesh_2D(node = True)
    post.plot_forces()
    post.plot_diagram()
    post.plot_disp(scale=1e0, r=150, dir='y', pic=False, path="./")
    plt.show()
    """
    post.plot_stress(s='sx')
    post.plot_stress(s='sf')
    post.plot_stress(s='ty')
    post.plot_stress(s='svm')
    
    """
    return

def test_remaillage():
    mesh = Mesh(2, debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([0, 10])
    mesh.add_node([10, 10])
    mesh.add_node([20, 10])
    mesh.add_element([1, 2], "barre", "b", 15, 15, 5)
    mesh.add_element([2, 3], "barre", "b", 15, 15, 5)
    mesh.add_element([3, 1], "bracon", "r", 10, 10, 4)
    mesh.add_element([1, 4], "semelle", "g", 10, 10, 3)
    mesh.plot_mesh()
    mesh.maillage()
    mesh.plot_mesh(ex=True)
    plt.show()


if __name__ == "__main__":
    test_cantilever()

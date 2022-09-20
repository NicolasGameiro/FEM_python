#from log import logger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = (8, 6)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
from matplotlib.patches import Rectangle, Polygon
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class Plot():
    def __init__(self, res, mesh, model, show=False):
        self.show = show
        self.res = res
        self.mesh = mesh
        self.model = model

        self.fig, self.ax = plt.subplots()

    # ----- 2D -----

    def plot(self, node=True, loads=True, supports=True, diagram=False, deformed=False, save = False):
        self._plot_elements_2d()

        if node == True :
            self._plot_nodes_2d()

        if loads == True :
            self._plot_loads_2d()

        pass

    def _plot_elements_2d(self):

        node_list = self.res['node']
        element_list = self.res['element']

        for i in range(len(element_list)):
            xi, xj = node_list[element_list[i, 0] - 1, 0], node_list[element_list[i, 1] - 1, 0]
            yi, yj = node_list[element_list[i, 0] - 1, 1], node_list[element_list[i, 1] - 1, 1]
            self.ax.plot([xi, xj], [yi, yj], color='k', lw=1, linestyle='--', label="undeformed")

    def _plot_nodes_2d(self):
        """ Method to plot nodes in 2d

        :return:
        """
        # get node list from res
        node_list = self.res['node']

        # get x and y coordinate for each node
        x = [x for x in node_list[:, 0]]
        y = [y for y in node_list[:, 1]]

        size = 500
        offset = size / 40000.
        self.ax.scatter(x, y, c='y', s=size, zorder=5)
        for i, location in enumerate(zip(x, y)):
            self.ax.annotate(i + 1, (location[0] - offset, location[1] - offset), zorder=10)

    def _plot_loads_2d(self, type='nodal'):
        F = self.res['F']
        NL = self.res['node']
        EL = self.res['element']
        scale_force = np.max(np.abs(F))

        ### Trace les efforts

        if type == 'nodal':
            self.ax.quiver(NL[:, 0] - F[:, 0] / scale_force,
                           NL[:, 1] - F[:, 1] / scale_force,
                           F[:, 0],
                           F[:, 1],
                           color='r', angles='xy', scale_units='xy', scale=scale_force)
            max_force = np.max(F)
            dof = 2
            node = NL[0, :]
            val = F[0]
            small = 0.1
            def_scale = 1
            self._plot_2d_nodal_load(max_force, dof, node, val, small)

        elif type == 'dist':
            for elem in self.dist_load[1:]:
                pt1 = NL[elem[0] - 1]
                pt2 = NL[elem[1] - 1]
                self._plot_2d_distributed_load(pt1, pt2, elem[2])

        self.ax.grid()
        self.ax.set_ylim([-1, max(NL[:, 0])])
        self.ax.set_xlim([-1, max(NL[:, 1])])
        self.ax.axis('equal')

    def interpol(self, x1, x2, y1, y2, y3, y4, r):
        x3 = x1
        x4 = x2
        V = np.array([[1, x1, x1 ** 2, x1 ** 3],
                      [1, x2, x2 ** 2, x2 ** 3],
                      [0, 1, 2 * x3, 3 * x3 ** 2],
                      [0, 1, 2 * x4, 3 * x4 ** 2]])
        # print(V)
        R = np.array([y1, y2, y3, y4])
        R = np.vstack(R)
        P = np.hstack(np.linalg.inv(V).dot(R))
        P = P[::-1]
        p = np.poly1d([x for x in P])
        x = np.linspace(x1, x2, r)
        y = p(x)
        return x, y

    def plot_disp_f_ex(self, scale=1e4, r=150):
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        x_scatter = []
        y_scatter = []
        color = []
        plt.figure()
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            plt.plot([xi, xj], [yi, yj], color='k', lw=1, linestyle='--')
        for i in range(len(EL)):
            x1 = NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 3] * scale
            x2 = NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 3] * scale
            y1 = NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 3 + 1] * scale
            y2 = NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 3 + 1] * scale
            y3 = U[(EL[i, 0] - 1) * 3 + 2]
            y4 = U[(EL[i, 1] - 1) * 3 + 2]
            L_e = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            c = np.round((x2 - x1) / L_e, 2)
            # print("c =", c)
            a = np.arccos(c) % 1
            # print("a = ", a)
            x, y = self.interpol(x1[0], x2[0], y1[0], y2[0], y3[0] + a, -y4[0] + a, r)
            x_scatter.append(x)
            y_scatter.append(y)
            color.append(np.linspace(U[(EL[i, 0] - 1) * 3 + 1], U[(EL[i, 1] - 1) * 3 + 1], r))
        # Permet de reverse la barre de couleur si max negatif
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        self.ax.scatter(x_scatter, y_scatter, c=color, cmap=cmap, s=10, edgecolor='none')
        self.ax.colorbar(label='disp'
                     , orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        return

    def plot_diagram(self, type="M"):
        """ Plot diagram on the mesh

        :param type: choose the type of diagram to plot (options : M)
        :type type: str
        :return:
        """
        Reactions = self.res["f"]
        NL, EL = self.mesh.node_list, self.mesh.element_list
        fig, ax = plt.subplots(3)

        self._plot_elements_2d()

        for elem in EL:
            ni, nj = elem
            xi, xj = NL[ni - 1][0], NL[nj - 1][0]
            yi, yj = NL[ni - 1][1], NL[nj - 1][1]
            ### T/C
            ri, rj = Reactions[(ni - 1),0], Reactions[(nj - 1),0]
            ax[0].plot([xi, xi, xj, xj], [0, ri, rj, 0], 'r', linewidth = 2)
            ax[0].fill([xi, xi, xj, xj], [0, ri, rj, 0], 'c', alpha=0.3)
            ### Shear
            ri, rj = -Reactions[(ni - 1),1], Reactions[(nj - 1), 1]
            ax[1].plot([xi, xi, xj, xj], [0, ri, rj, 0], 'r', linewidth = 2)
            ax[1].fill([xi, xi, xj, xj], [0, ri, rj, 0], 'c', alpha=0.3)
            ### Bending moment
            ri, rj = -Reactions[(ni - 1), 2], Reactions[(nj - 1), 2]
            ax[2].plot([xi, xi, xj, xj], [0, ri, rj, 0], 'r', linewidth = 2)
            ax[2].fill([xi, xi, xj, xj], [0, ri, rj, 0], 'c', alpha=0.3)
        '''
        x = [xi, xi, xj, xj, xi]
        y = [yi, yi + ri, yj - rj, yj, yi]
        ax.add_patch(Polygon(xy=list(zip(x, y)), fill=True, color="r", alpha=0.5, lw=0))
        '''

        return

    def plot_disp(self, scale=1e2, r=150, dir='x', pic=False, path="./"):
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.res['U']
        x_scatter = []
        y_scatter = []
        color = []

        # maillage non deforme
        self._plot_elements_2d()

        for i in range(len(EL)):
            if dir == 'y':
                plt.title("y")
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0], r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 3 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 3 + 1] * scale, r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 3 + 1], U[(EL[i, 1] - 1) * 3 + 1], r))
            elif dir == "x":
                plt.title("x")
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 3] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 3] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 3], U[(EL[i, 1] - 1) * 3], r))
            elif dir == "sum":
                plt.title("sum")
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 3] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 3] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 3 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 3 + 1] * scale, r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 3] + U[(EL[i, 0] - 1) * 3 + 1],
                                         U[(EL[i, 1] - 1) * 3] + U[(EL[i, 1] - 1) * 3 + 1], r))
        # Permet de reverse la barre de couleur si max negatif
        # TODO : a corriger
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        self.ax.scatter(x_scatter, y_scatter, c=color, cmap=cmap, s=10, edgecolor='none')
        plt.colorbar(label='disp'
                     , orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        if pic:
            plt.savefig(path + 'res_' + dir + '.png', format='png', dpi=200)
        return

    def plot_stress(self, scale=1e1, r=100, s='sx', pic=False, path="./"):
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.res['U']
        S = self.res['stress']
        x_scatter = []
        y_scatter = []
        color = []
        plt.figure()
        # maillage non deforme
        self._plot_elements_2d()
        # Maillage deforme avec coloration en fonction de la contrainte
        for i in range(len(EL)):
            n1, n2 = EL[i, 0] - 1, EL[i, 1] - 1
            if s == 'sx':
                plt.title("tensile stress (sx)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5])
            elif s == "sf":
                plt.title("bending stress (sf)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5 + 1])
            elif s == "ty":
                plt.title("shear stress (ty)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5 + 2])
            elif s == "svm":
                plt.title("Von Mises Stress (svm)")
                x_scatter.append(np.linspace(NL[n1, 0] + U[n1 * 3] * scale, NL[n2, 0] + U[n2 * 3] * scale, r))
                y_scatter.append(np.linspace(NL[n1, 1] + U[n1 * 3 + 1] * scale, NL[n2, 1] + U[n2 * 3 + 1] * scale, r))
                color.append(np.ones(r) * S[i * 5 + 3])
        # Permet de reverse la barre de couleur si max negatif
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        self.ax.scatter(x_scatter, y_scatter, c=color, cmap=cmap, s=10, edgecolor='none')
        self.ax.colorbar(label='stress', orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.axis('equal')
        if pic:
            plt.savefig(path + 'stress_' + s + '.png', format='png', dpi=200)
        return

    def _plot_element_axis(self, elem):
        print(elem)
        NL = self.mesh.node_list
        node_i = NL[elem[0] - 1]
        node_j = NL[elem[1] - 1]
        dx, dy, dz = node_j[0] - node_i[0], node_j[1] - node_i[1], node_j[2] - node_i[2]
        vx = [dx, dy, dz]  # vecteur directeur de l'element
        RR = self.Rot_3D(vx)
        rr = RR[0:3, 0:3]
        print(rr)
        if True in np.isnan(rr):
            vy = [0, 1, 0]
            vz = [0, 0, 1]
        else:
            vy = rr * [0, 1, 0]
            vz = rr * [0, 0, 1]

        self.ax.quiver(node_i[0], node_i[1], node_i[2], vx[0], vx[1], vx[2], color='r', length=0.1, normalize=True)
        self.ax.quiver(node_i[0], node_i[1], node_i[2], vy[0], vy[1], vy[2], color='g', length=0.1, normalize=True)
        self.ax.quiver(node_i[0], node_i[1], node_i[2], vz[0], vz[1], vz[2], color='b', length=0.1, normalize=True)

    def _plot_2d_nodal_load(self, max_force, dof, node, val, small):
        """Plots a graphical representation of a nodal force. A straight arrow is plotted for a
        translational load and a curved arrow is plotted for a moment.

        :param ax: Axes object on which to plot
        :type ax: :class:`matplotlib.axes.Axes`
        :param max_force: Maximum translational nodal load
        :type max_force: float
        :param dof: degre of freedom of the load applied
        :type dof: int
        :param node: coordinate of the node
        :type node: np.array
        :param val: load value at the node
        :type val: float
        :param small: A dimension used to scale the support
        :type small: float
        """

        x = node[0]
        y = node[1]

        if max_force == 0:
            val = 1
        else:
            val = val / max_force

        offset = 0.5 * small
        (angle, num_el) = (0, 1)
        s = np.sin(angle * np.pi / 180)
        c = np.cos(angle * np.pi / 180)

        # plot nodal force
        if dof in [0, 1]:
            lf = abs(val) * 1.5 * small  # arrow length
            lh = 0.6 * small  # arrow head length
            wh = 0.6 * small  # arrow head width
            lf = max(lf, lh * 1.5)

            n = np.array([c, s])
            inward = (n[dof] == 0 or np.sign(n[dof]) == np.sign(val))

            to_rotate = (dof) * 90 + (n[dof] > 0) * 180
            sr = np.sin(to_rotate * np.pi / 180)
            cr = np.cos(to_rotate * np.pi / 180)
            rot_mat = np.array([[cr, -sr], [sr, cr]])

            ll = np.array([[offset, offset + lf], [0, 0]])
            p0 = offset + (not inward) * lf
            p1 = p0 + (inward) * lh - (not inward) * lh
            pp = np.array([[p1, p1, p0], [-wh / 2, wh / 2, 0]])

            # correct end of arrow line
            if inward:
                ll[0, 0] += lh
            else:
                ll[0, 1] -= lh

            rl = np.matmul(rot_mat, ll)
            rp = np.matmul(rot_mat, pp)
            rp[0, :] += x
            rp[1, :] += y
            s = 0
            e = None

        # plot nodal moment
        else:
            lh = 0.4 * small  # arrow head length
            wh = 0.4 * small  # arrow head width
            rr = 1.5 * small
            ths = np.arange(100, 261)
            rot_mat = np.array([[c, -s], [s, c]])

            # make arrow tail around (0,0)
            ll = np.array([rr * np.cos(ths * np.pi / 180), rr * np.sin(ths * np.pi / 180)])

            # make arrow head at (0,0)
            pp = np.array([[-lh, -lh, 0], [-wh / 2, wh / 2, 0]])

            # rotate arrow head around (0,0)
            if val > 0:
                thTip = 90 - ths[11]
                xTip = ll[:, -1]
                s = 0
                e = -1
            else:
                thTip = ths[11] - 90
                xTip = ll[:, 0]
                s = 1
                e = None

            cTip = np.cos(thTip * np.pi / 180)
            sTip = np.sin(thTip * np.pi / 180)
            rTip = np.array([[cTip, -sTip], [sTip, cTip]])
            pp = np.matmul(rTip, pp)

            # shift arrow head to tip
            pp[0, :] += xTip[0]
            pp[1, :] += xTip[1]

            # rotate arrow to align it with the node
            rl = np.matmul(rot_mat, ll)
            rp = np.matmul(rot_mat, pp)
            rp[0, :] += x
            rp[1, :] += y

        self.ax.plot(rl[0, s:e] + x, rl[1, s:e] + y, 'r-', linewidth=2)
        self.ax.add_patch(Polygon(np.transpose(rp), facecolor='r'))

    def _plot_2d_distributed_load(self, pt1, pt2, q):
        """ Method to plot a distributed load in 2d

        :param pt1:
        :param pt2:
        :param q:
        :return:
        """
        x1, x2 = pt1[0], pt2[0]
        y1, y2 = pt1[1], pt2[1]
        dx, dy = x2 - x1, y2 - y1
        L = np.sqrt(dx ** 2 + dy ** 2)
        a = np.arctan(dy / dx)

        nb_pt = 5
        amplitude = 1
        x = np.linspace(pt1[0], pt2[0], nb_pt)
        y = np.linspace(pt1[1], pt2[1], nb_pt)

        for i in range(0, nb_pt):
            self.ax.arrow(x[i],  # x1
                          y[i] + amplitude,  # y1
                          0,  # x2 - x1
                          -amplitude,  # y2 - y1
                          color='r',
                          lw=1,
                          length_includes_head=True,
                          head_width=0.02,
                          head_length=0.05,
                          zorder=6)
        self.ax.plot([pt1[0], pt2[0]], [pt1[1] + 1, pt2[1] + 1], lw=1, color='r', zorder=6)
        self.ax.text(x1 + dx / 2 * 0.9, y1 + dy / 2 + amplitude * 1.2,
                     "q = " + str(q / 1000) + " kN/m",
                     size=10, zorder=2, color="k")
        x = [pt1[0], pt2[0], pt2[0], pt1[0], pt1[0]]
        y = [pt1[1], pt2[1], pt2[1] + 1, pt1[1] + 1, pt1[1]]

        self.ax.add_patch(Polygon(xy=list(zip(x, y)), fill=True, color='red', alpha=0.1, lw=0))
        return

    # ----- 3D -----

    def plot_mesh_3D(self, size=50):
        NL = self.res['node']
        EL = self.res['element']
        x = [x for x in NL[:, 0]]
        y = [y for y in NL[:, 1]]
        z = [z for z in NL[:, 2]]
        self.ax.scatter(x, y, z, c='y', s=size, zorder=1)
        for i, location in enumerate(zip(x, y)):
            self.ax.text(x[i], y[i], z[i], str(i + 1), size=20, zorder=2, color="k")
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            zi, zj = NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2]
            line, = self.ax.plot([xi, xj], [yi, yj], [zi, zj], color=self.mesh.color[i], lw=1, linestyle='--')
            line.set_label(self.mesh.name[i])

    def charge_3D(self, ax, pt1, pt2, q):
        x1, x2 = pt1[0], pt2[0]
        y1, y2 = pt1[1], pt2[1]
        z1, z2 = pt1[2], pt2[2]
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # a = np.arctan(dy / dx)
        nb_pt = 10
        amplitude = 1
        x = np.linspace(x1, x2, nb_pt)
        y = np.linspace(y1, y2, nb_pt)
        z = np.linspace(z1, z2, nb_pt)

        for i in range(0, nb_pt):
            a = Arrow3D([x[i], x[i]],
                        [y[i], y[i]],
                        [z[i] + amplitude, z[i]],
                        mutation_scale=10,
                        lw=2, arrowstyle="-|>", color="r")
            self.ax.add_artist(a)
        line, = ax.plot([x1, x2], [y1, y2], [z1 + amplitude, z2 + amplitude], color='r', lw=1)
        self.ax.text(x1 + dx / 2, y1 + dy / 2, z1 + dz / 2,
                "q = " + str(q) + " kN/m",
                size=20, zorder=2, color="k")
        return

    def plot_forces3D(self, type='nodal', pic=False, path="./"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        # plt.gca(projection='3d')
        F = self.res['F']
        NL = self.res['node']
        EL = self.res['element']
        scale_force = np.max(np.abs(F))

        # Affichage du maillage

        self.plot_mesh_3D(ax=ax)

        ### Trace les efforts
        if type == 'nodal':
            f_length = np.sqrt(F[:, 0] ** 2 + F[:, 1] ** 2 + F[:, 2] ** 2) / scale_force
            plt.quiver(NL[:, 0] - F[:, 0] / scale_force, NL[:, 1] - F[:, 1] / scale_force,
                       NL[:, 2] - F[:, 2] / scale_force,
                       F[:, 0] / scale_force, F[:, 1] / scale_force, F[:, 2] / scale_force, color='r', pivot="tail",
                       length=max(f_length), normalize=True)
        elif type == 'dist':
            for elem in self.model.dist_load[1:]:
                pt1 = NL[elem[0] - 1]
                pt2 = NL[elem[1] - 1]
                self.charge_3D(ax, pt1, pt2, elem[2])
        # self.charge_3D(ax, [0, 0, 2.5], [0, 6 / 2, 5], 1)
        ax.set_title("Structure")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        ax.view_init(elev=20., azim=-20.)
        """
        x, y , z = 0, 0, 2.5+1
        u, v, w = 0, 0, -1
        ax.quiver(x, y, z, u, v, w, length=1, normalize=True)
        """
        ax.set_xlim(-1, max(NL[:, 0]) + 1)
        ax.set_ylim(-1, max(NL[:, 1]) + 1)
        ax.set_zlim(0, max(NL[:, 2]) + 1)
        plt.tight_layout()
        plt.grid()
        if pic:
            plt.savefig(path + 'load.png', format='png', dpi=200)
        return ax

    def plot_disp_f_3D(self, scale=1e0, r=80, dir='x', pic=False, path="./"):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        U = self.U
        x_scatter = []
        y_scatter = []
        z_scatter = []
        color = []
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            zi, zj = NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2]
            line, = ax.plot([xi, xj], [yi, yj], [zi, zj], color=self.mesh.color[i], lw=1, linestyle='--')
            line.set_label(self.mesh.name[i])
            self._plot_element_axis(EL[i, :])
        for i in range(len(EL)):
            if dir == 'y':
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0], r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 6 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 6 + 1] * scale, r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6 + 1], U[(EL[i, 1] - 1) * 6 + 1], r))
            elif dir == "x":
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 6] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 6] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1], r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6], U[(EL[i, 1] - 1) * 6], r))
            elif dir == "z":
                x_scatter.append(
                    np.linspace(NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 6] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1], r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2] + U[(EL[i, 0] - 1) * 6 + 2] * scale,
                                             NL[EL[i, 1] - 1, 2] + U[(EL[i, 1] - 1) * 6 + 2], r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6 + 2], U[(EL[i, 1] - 1) * 6 + 2], r))
            elif dir == "sum":
                x_scatter.append(np.linspace(NL[EL[i, 0] - 1, 0] + U[(EL[i, 0] - 1) * 6] * scale,
                                             NL[EL[i, 1] - 1, 0] + U[(EL[i, 1] - 1) * 6] * scale, r))
                y_scatter.append(np.linspace(NL[EL[i, 0] - 1, 1] + U[(EL[i, 0] - 1) * 6 + 1] * scale,
                                             NL[EL[i, 1] - 1, 1] + U[(EL[i, 1] - 1) * 6 + 1] * scale, r))
                z_scatter.append(np.linspace(NL[EL[i, 0] - 1, 2] + U[(EL[i, 0] - 1) * 6 + 2] * scale,
                                             NL[EL[i, 1] - 1, 2] + U[(EL[i, 1] - 1) * 6 + 2] * scale, r))
                color.append(np.linspace(U[(EL[i, 0] - 1) * 6] + U[(EL[i, 0] - 1) * 6 + 1] + U[(EL[i, 0] - 1) * 6 + 2],
                                         U[(EL[i, 1] - 1) * 6] + U[(EL[i, 1] - 1) * 6 + 1] + U[(EL[i, 1] - 1) * 6 + 2],
                                         r))
        # Permet de reverse la barre de couleur si max negatif
        if min(U) > 0:
            cmap = plt.get_cmap('jet')
        elif min(U) <= 0:
            cmap = plt.get_cmap('jet_r')
        scat = ax.scatter3D(x_scatter, y_scatter, z_scatter, c=color, cmap=cmap, s=40, edgecolor='none')
        # ax.colorbar(label='disp', orientation='vertical')  # ScalarMappable(norm = norm_x, cmap = cmap ))
        plt.colorbar(scat)
        ax.set_title("Déplacement " + dir)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_box_aspect([1, 1, 1])
        self.set_equal_aspect_3D(ax)
        plt.tight_layout()
        plt.grid()
        if pic:
            plt.savefig(path + 'res_' + dir + '.png', format='png', dpi=200)
        return

    def set_equal_aspect_3D(self, ax):
        """
        Set aspect ratio of plot correctly
        Args:
            :ax: (obj) axis object
        """

        # See https://stackoverflow.com/a/19248731
        # ax.set_aspect('equal') --> raises a NotImplementedError
        # See https://github.com/matplotlib/matplotlib/issues/1077/

        extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

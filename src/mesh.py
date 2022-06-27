from prettytable import PrettyTable as pt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.figsize"] = (8, 6)
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
from matplotlib.patches import Rectangle, Polygon
from log import logger


class Mesh:
    def __init__(self, dim: int, node_list=[], element_list=[], ax = None, debug=False):
        """ Initiatiolisation d'un maillage à partir de sa dimension"""
        logger.info("Meshing...")
        logger.info(f"Mesh dimension : {dim}D")
        self.dim = dim
        self.node_list = np.empty((0, dim))
        self.node_list_ex = np.empty((0, dim))
        self.element_list = np.empty((0, 2), dtype=int)
        self.element_list_ex = np.empty((0, 2), dtype=int)
        self.name = np.empty((0, 1))
        self.color = np.empty((0, 1))
        self.Section = np.empty((0, 2))
        self.name_ex = np.empty((0, 1))
        self.color_ex = np.empty((0, 1))
        self.Section_ex = np.empty((0, 2))
        self.div = np.empty((0, 1), dtype=int)
        self.debug = debug
        if ax == None :
            self.figure_axis = plt.figure(figsize=(8, 6)).add_subplot(111)
        else :
            self.figure_axis = ax

    def add_node(self, node: list):
        """ Ajout un noeud au maillage """

        if len(node) != self.dim:
            print("Erreur : format du noeud incorrect")
            logger.info("Erreur : format du noeud incorrect")
        else:
            found, index = self.check_node(node)
            if found == False:
                self.node_list = np.append(self.node_list, np.array([node]), axis=0)
                # print("noeud ajouté")
                logger.info(f"noeud {node} ajouté")
            else:
                print("noeud deja dans le maillage")
                logger.info("noeud deja dans le maillage")
        if self.debug == True:
            print(self.node_list)

    def check_node(self, node):
        """Verification qu'un noeud n'est pas déjà dans le maillage"""
        index = -1
        found = False
        while (found is not True) and (index + 1 < len(self.node_list)) and (self.node_list.size != 0):
            index += 1
            if self.dim == 2:
                if (self.node_list[index][0] == node[0]) and (self.node_list[index][1] == node[1]):
                    found = True
            elif self.dim == 3:
                if (self.node_list[index][0] == node[0]) and (self.node_list[index][1] == node[1]) and (
                        self.node_list[index][2] == node[2]):
                    found = True
        return found, index

    def check_node_ex(self, node):
        index = -1
        found = False
        while (found is not True) and (index + 1 < len(self.node_list_ex)) and (self.node_list_ex.size != 0):
            index += 1
            if self.dim == 2:
                if (self.node_list_ex[index][0] == node[0]) and (self.node_list_ex[index][1] == node[1]):
                    found = True
            elif self.dim == 3:
                if (self.node_list_ex[index][0] == node[0]) and (self.node_list_ex[index][1] == node[1]) and (
                        self.node_list_ex[index][2] == node[2]):
                    found = True
        return found, index

    def del_node(self, node):
        if len(node) != self.dim:
            print("Erreur : format du noeud incorrect")
            logger.info("Erreur : format du noeud incorrect")
        else:
            found, index = self.check_node(node)
            if found == True:
                self.node_list = np.delete(self.node_list, index, 0)
                print("noeud supprimé")
                logger.info(f"noeud {node} supprimé")
            else:
                print("noeud non trouvé")
            if self.debug == True:
                print(self.node_list)

    def reset_node(self):
        self.node_list = np.array([])
        print("liste des noeuds vidée")
        logger.info("suppression de la liste des noeuds")
        if self.debug == True:
            print(self.node_list)
        return

    ### GESTION DES ELEMENTS

    def check_elem(self, elem):
        index = -1
        found = False
        while (found is not True) and (index + 1 < len(self.element_list)) and (self.element_list.size != 0):
            index += 1
            if (self.element_list[index][0] == elem[0]) and (self.element_list[index][1] == elem[1]):
                found = True
        return found, index

    def add_element(self, elem, name="poutre", color="k", h="22", l="10", div=1):
        found, index = self.check_elem(elem)
        if found == False:
            self.element_list = np.append(self.element_list, np.array([elem]), axis=0)
            self.name = np.append(self.name, np.array(name))
            self.color = np.append(self.color, np.array(color))
            self.Section = np.append(self.Section, np.array([[h, l]]), axis=0)
            self.div = np.append(self.div, np.array(div))
            # print("element ajouté")
        else:
            print("element deja dans le maillage")
        if self.debug == True:
            print(self.element_list)
            print(self.name)
            print(self.color)
            print(self.Section)

    def del_element(self, element):
        if len(element) != self.dim:
            print("Erreur : format de l'element incorrect")
        else:
            found, index = self.check_elem(element)
            if found == True:
                self.element_list = np.delete(self.element_list, index, 0)
                print("element supprimé")
            else:
                print("element non trouvé")
            if self.debug == True:
                print(self.element_list)

    def add_section(self, S):
        self.S_list = np.append(self.S_list, [S], axis=0)

    def node_table(self):
        tab = pt()
        if self.dim == 2:
            tab.field_names = ["Node", "X", "Y"]
            for i in range(len(self.node_list)):
                tab.add_row([int(i + 1), self.node_list[i, 0], self.node_list[i, 1]])
        else:
            tab.field_names = ["Node", "X", "Y", "Z"]
            for i in range(len(self.node_list)):
                tab.add_row([int(i + 1), self.node_list[i, 0], self.node_list[i, 1], self.node_list[i, 2]])
        print(tab)

    def maillage(self):
        #TODO : créer une table de correspondance entre le maillage maitre et eleve
        """Fonction permettant de remailler le maillage initial"""
        last_node = 1
        for i in range(len(self.element_list)):
            el = self.element_list[i]
            n1, n2 = el[0] - 1, el[1] - 1
            div = self.div[i]
            x1, y1 = self.node_list[n1]
            x2, y2 = self.node_list[n2]
            vecteur_directeur = np.array([x2 - x1, y2 - y1])
            for j in range(div + 1):
                node = [x1 + vecteur_directeur[0] / div * j,
                        y1 + vecteur_directeur[1] / div * j]
                found, index = self.check_node_ex(node)
                #print(found, node)
                if found == False:
                    self.node_list_ex = np.append(self.node_list_ex, np.array([node]), axis=0)
                    #print('ajout du noeud ', len(self.node_list_ex))
                    # ajout du nouvel element
                    if len(self.node_list_ex) > 1:
                        self.element_list_ex = np.append(self.element_list_ex,
                                                         np.array([[last_node, len(self.node_list_ex)]]), axis=0)
                        self.name_ex = np.append(self.name_ex, np.array([self.name[i]]))
                        self.color_ex = np.append(self.color_ex, np.array([self.color[i]]))
                        self.Section_ex = np.append(self.Section_ex, np.array([self.Section[i]]), axis=0)
                        #print("ajout de l'element ", self.element_list_ex[-1])
                        last_node = last_node + 1
                else:
                    #print((node[0] == x1), (node[1] == y1))
                    if (not (node[0] == x1)) or (not (node[1] == y1)):
                        n = np.where(self.node_list_ex == node)
                        n = max(list(n[0]), key=list(n[0]).count)
                        #print("le noeud ", n + 1, "existe deja")
                        self.element_list_ex = np.append(self.element_list_ex, np.array([[last_node, n + 1]]), axis=0)
                        self.name_ex = np.append(self.name_ex, np.array([self.name[i]]))
                        self.color_ex = np.append(self.color_ex, np.array([self.color[i]]))
                        self.Section_ex = np.append(self.Section_ex, np.array([self.Section[i]]), axis=0)
                        #print("ajout de l'element ", self.element_list_ex[-1])
            #print('\n #######')

        nb_elem = sum(self.div)
        #print("new nb_elem =", nb_elem)

        found, index = self.check_node_ex(node)
        #print(found, node)
        if found == False:
            pend(self.node_list_ex, [np.array(self.node_list[self.element_list[-1][1] - 1])], axis=0)
        #print("node list extended : ", self.node_list_ex)
        #print("element list extended :", self.element_list_ex)
        #print("section extended list :", self.Section_ex)

        # Si il n'y a pas de subdivision des éléments ont garde le maillage d'origine
        test = np.count_nonzero(self.div - np.ones(len(self.div)))
        #print(test)
        if test == 0:
            #print("pas de maillage subdivisé", self.div)
            self.node_list_ex, self.element_list_ex, self.name_ex, self.color_ex, self.Section_ex = self.node_list, self.element_list, self.name, self.color, self.Section

        return self.node_list_ex, self.element_list_ex, self.name_ex, self.color_ex, self.Section_ex

    def __str__(self):
        return f""" Information sur le maillage : \n
    - Nombre de noeuds : {len(self.node_list)}\n 
    - Nombre d'éléments : {len(self.element_list)}
    """

    def plot_mesh(self, ex=False, pic=False, path="./"):
        self.figure_axis = plt.figure(figsize=(8, 6)).add_subplot(111)
        if ex == True:
            NL = self.node_list_ex
            EL = self.element_list_ex
            name = self.name_ex
            color = self.color_ex
            section = self.Section_ex
        else:
            NL = self.node_list
            EL = self.element_list
            name = self.name
            color = self.color
            section = self.Section
        if self.dim == 2:
            self.figure_axis, pts = self.geom2D(NL, EL, name, color, section, pic)
        else:
            self.figure_axis = self.geom3D(NL, EL, name, color, section, pic)
        return self.figure_axis, pts

    def geom2D(self, NL, EL, name, color, section, pic=False, path="./"):
        ax = self.figure_axis
        x = [x for x in NL[:, 0]]
        y = [y for y in NL[:, 1]]
        size = 10
        offset = size / 40000.
        pts = ax.scatter(x, y, c='k', marker="s", s=size, zorder=5)
        color_list = []
        for i, location in enumerate(zip(x, y)):
            ax.annotate(i + 1, (location[0] - offset, location[1] - offset), zorder=10)
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            ax.plot([xi, xj], [yi, yj], color=color[i], lw=2, linestyle='--',
                    label=name[i] if color[i] not in color_list else '')

            h = section[i][0] / 100

            if xi != xj:
                pt1 = [xi, yi - h / 2]
                pt2 = [xj, yj - h / 2]
                pt3 = [xj, yj + h / 2]
                pt4 = [xi, yi + h / 2]
            else:
                pt1 = [xi - h / 2, yi]
                pt2 = [xj - h / 2, yj]
                pt3 = [xj + h / 2, yj]
                pt4 = [xi + h / 2, yi]

            x = pt1[0], pt2[0], pt3[0], pt4[0], pt1[0]
            y = pt1[1], pt2[1], pt3[1], pt4[1], pt1[1]
            ax.add_patch(Polygon(xy=list(zip(x, y)), color=color[i], fill=True, alpha=0.3, lw=0))
            # pour verifier que la legende n'existe pas deja
            if (color == color[i]).sum() > 1:
                color_list.append(color[i])
        ax.axis('equal')
        ax.legend()
        plt.grid()
        if pic:
            plt.savefig(path + 'geom.png', format='png', dpi=200)
        return ax, pts

    def geom3D(self, NL, EL, name, color, section, pic=False, path="./"):
        fig = plt.figure(figsize=(8, 6))
        # plt.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        x = [x for x in NL[:, 0]]
        y = [y for y in NL[:, 1]]
        z = [z for z in NL[:, 2]]
        ax.scatter(x, y, z, c='y', s=100, zorder=1)
        for i, location in enumerate(zip(x, y)):
            ax.text(x[i], y[i], z[i], str(i + 1), size=20, zorder=2, color="k")
        for i in range(len(EL)):
            xi, xj = NL[EL[i, 0] - 1, 0], NL[EL[i, 1] - 1, 0]
            yi, yj = NL[EL[i, 0] - 1, 1], NL[EL[i, 1] - 1, 1]
            zi, zj = NL[EL[i, 0] - 1, 2], NL[EL[i, 1] - 1, 2]
            line, = ax.plot([xi, xj], [yi, yj], [zi, zj], color='k', lw=1, linestyle='--')
            line.set_label('undeformed')
        ax.set_title("Structure")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.view_init(elev=20., azim=-20.)
        ax.set_box_aspect([1, 1, 1])
        if pic:
            plt.savefig(path + 'geom.png', format='png', dpi=200)
        return fig


if __name__ == "__main__":
    mesh = Mesh(2, debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([0, 10])  # inches
    mesh.add_node([10, 10])  # inches
    mesh.add_node([20, 10])
    mesh.add_element([1, 2], "barre", "b", 15, 15, 5)
    mesh.add_element([2, 3], "barre", "b", 15, 15, 5)
    mesh.add_element([3, 1], "bracon", "r", 10, 10, 4)
    mesh.add_element([1, 4], "semelle", "g", 10, 10, 3)
    mesh.plot_mesh()
    mesh.maillage()
    mesh.plot_mesh(ex=True)
    plt.show()

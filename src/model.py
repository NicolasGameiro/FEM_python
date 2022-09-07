# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:41:46 2022

@author: ngameiro
"""

"""[Summary]

:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
:type [ParamName]: [ParamType](, optional)
...
:raises [ErrorType]: [ErrorDescription]
...
:return: [ReturnDescription]
:rtype: [ReturnType]
"""

# Model
import numpy as np
import math
from prettytable import PrettyTable as pt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from mesh import Mesh


class FEM_Model():
    def __init__(self, mesh, E=2.1E9):
        self.mesh = mesh
        self.E = E
        if self.mesh.dim == 2:
            self.mesh.node_list, self.mesh.element_list, self.mesh.name, self.mesh.color, self.mesh.Section = self.mesh.maillage()
            self.load = np.zeros([len(self.mesh.node_list), 3])
            self.bc = np.eye(len(self.mesh.node_list) * 3)
            self.U = np.zeros(len(self.mesh.node_list) * 3)
            self.React = np.zeros(len(self.mesh.node_list) * 3)
            self.S = np.empty((0, 5))
        elif self.mesh.dim == 3:
            self.load = np.zeros([len(self.mesh.node_list), 6])
            self.bc = np.eye(len(self.mesh.node_list) * 6)
            self.U = np.zeros(len(self.mesh.node_list) * 6)
            self.React = np.zeros(len(self.mesh.node_list) * 6)
            self.S = np.empty((0 , 7))
        self.dist_load = np.array([[1, 2, 0]])
        self.lbc = []

    def apply_load(self, node_load, node_index):
        """ Method to apply a load at a specified node

        :param node_load: value of the load
        :type node_load: float
        :param node_index: index of the node in the mesh
        :type node_index: int
        :return:
        """

        if node_index > len(self.mesh.node_list):
            print("Error : node specified not in the mesh")
        elif (len(node_load) == 3) or (len(node_load) == 6):
            self.load[node_index - 1, :] = node_load
            # print("nodal load applied")
            if self.mesh.debug == True:
                print(self.load)
        else:
            print("Error : uncorrect load format")

    def get_nodes_loaded(self):
        """ Method to get the list of the index of loaded nodes

        :return: list of loaded nodes
        """
        nodes_loaded = []
        for node_index, row in enumerate(self.load):

            # Test if row is not all zeros
            if row.any():
                print("Node %s is loaded" % node_index)
                nodes_loaded.append(node_index)

        return nodes_loaded

    def apply_distributed_load(self, q, element):
        L = self.get_length(element)
        if self.mesh.dim == 2:
            Q = np.array([0,
                          -q * L / 2,
                          -q * L ** 2 / 12,
                          0,
                          -q * L / 2,
                          q * L ** 2 / 12])
            self.load[element[0] - 1] = self.load[element[0] - 1] + Q[:3]
            self.load[element[1] - 1] = self.load[element[1] - 1] + Q[3:6]
        elif self.mesh.dim == 3:
            Q = np.array([0, 0, -q * L / 2,
                          -q * L ** 2 / 12, 0, 0,
                          0, 0, -q * L / 2,
                          q * L ** 2 / 12, 0, 0])
            self.load[element[0] - 1] = self.load[element[0] - 1] + Q[:6]
            self.load[element[1] - 1] = self.load[element[1] - 1] + Q[6:12]
        self.dist_load = np.append(self.dist_load, [[element[0], element[1], q]], axis=0)
        # print(self.dist_load)

    def apply_bc(self, node_bc, node_index):
        """ Method to apply boundary condition to a node

        :param node_bc: degree of freedom restrainted
        :type node_bc: list
        :param node_index: index of the node in the mesh
        :type node_index: int
        :return:
        """
        if node_index > len(self.mesh.node_list):
            print("Error : node specified not in the mesh")

        # For dimension 2
        elif len(node_bc) == 3:
            for i in range(len(node_bc)):
                if node_bc[i] == 1:
                    self.lbc.append(i + 3 * (node_index - 1))
            # print("boundary condition applied")

        # For dimension 3
        elif len(node_bc) == 6:
            for i in range(len(node_bc)):
                if node_bc[i] == 1:
                    self.lbc.append(i + 6 * (node_index - 1))
            # print("boundary condition applied")

        else:
            print("Error : uncorrect bc format")

    def get_2d_rotation_matrix(self, c, s):
        """ Calculate the rotation matrix in 2D

        :param c: cosinus
        :type c: float
        :param s: sinus
        :type s: float
        :return: the rotation matrix
        """
        Rotation_matrix = np.array([[c, -s, 0, 0, 0, 0],
                                    [s, c, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, c, -s, 0],
                                    [0, 0, 0, s, c, 0],
                                    [0, 0, 0, 0, 0, 1]])
        return Rotation_matrix

    def get_3d_rotation_matrix(self, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2

        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        RR = np.identity(12)
        vec1 = [1, 0, 0]
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        RR[0:3, 0:3] = rotation_matrix
        RR[3:6, 3:6] = rotation_matrix
        RR[6:9, 6:9] = rotation_matrix
        RR[9:12, 9:12] = rotation_matrix
        return RR

    def mini_rot(self, c, s):
        R = np.array([[c, s],
                      [-s, c]])
        return R

    def get_2d_element_stiffness_matrix(self, L_e, h, b):
        S = h * b  # * 1e-4
        I = b * h ** 3 / 12  # * 1e-8
        G = 1 #necessite le module de poisson
        k = 12 * self.E * I / G / S / L_e ** 2 # k = 0 : pas de cisaillement
        Ktc = S * self.E / L_e
        Kf1 = 12 * self.E * I / L_e ** 2 / (1 + k)
        Kf2 = 6 * self.E * I / L_e ** 2 / (1 + k)
        Kf3 = self.E * I * (2 - k) / L_e / (1 + k)
        Kf4 = self.E * I * (4 + k) / L_e / (1 + k)
        K_elem = np.array([[ Ktc,    0,    0, -Ktc,    0,    0],
                           [   0,  Kf1,  Kf2,    0, -Kf1,  Kf2],
                           [   0,  Kf2,  Kf4,    0, -Kf2,  Kf3],
                           [-Ktc,    0,    0,  Ktc,    0,    0],
                           [   0, -Kf1, -Kf2,    0,  Kf1, -Kf2],
                           [   0,  Kf2,  Kf3,    0, -Kf2,  Kf4]])
        return K_elem

    def get_2d_element_mass_matrix(self, L_e, rho, h, b):
        """ Method to get the mass matrix of an element in 2d

        :param L_e: length of the element
        :param rho: volumic mass of the element
        :param h: height of the element
        :param b: width of the element
        :return: element mass matrix
        """

        A = h * b
        coeff = rho * A * L_e / 420

        element_mass_matrix = coeff * np.array([[140, 0, 0, 70, 0, 0],
                                                [0, 156, 22 * L_e, 0, 54, -13 * L_e],
                                                [0, 22 * L_e, 4 * L_e ** 2, 0, 13 * L_e, -3 * L_e ** 2],
                                                [70, 0, 0, 140, 0, 0],
                                                [0, 54, 13 * L_e, 0, 156, -22 * L_e],
                                                [0, -13 * L_e, -3 * L_e ** 2, 0, -22 * L_e, 4 * L_e ** 2]])

        return element_mass_matrix

    def get_2d_element_mass_matrix(self, type=0):
        """

        :param type: str
        :return:
        """

        # get consistent mass matrix
        if type == 0:
            pass

        # get lumped mass matrix
        elif type == 1:
            pass

        return

    def stress_2(self):
        S = self.mesh.S
        I = self.mesh.Iy
        h = 0.22
        self.sig = np.zeros([len(self.mesh.node_list), 3])
        for i in range(len(self.mesh.node_list)):
            # en MPa
            self.sig[i, 0] = self.load[i, 0] / S / 1e6  # traction/compression (en MPa)
            self.sig[i, 1] = self.load[i, 1] / S / 1e6  # cisaillement (en MPa)
            self.sig[i, 2] = self.load[i, 2] / I * (h / 2) / 1e6  # flexion (en MPa)
        print(self.sig)

    def get_3d_element_stiffness_matrix(self, L: float, h: float, b: float, E: float = 1, nu: float = 0.3, ay: float = 0,
                                        az: float = 0) -> np.array:
        """ Calcul de la matrice de raideur avec prise en compte de l'énergie cisaillement avec les termes ay et az.

        :param L: longueur de l'element
        :type L: float
        :param E: Module d'Young
        :type E: float
        :param G: Module de coulomb
        :type G: float
        :param J: Module de torsion
        :type J: float
        :param ay:
        :type ay:
        :param az:
        :type az:
        :return: matrice de raideur en 3D
        :rtype: np.array
        """
        G = 1  # E/2/(1+nu)
        S = 1  # h * b
        Iy = 1  # b * h ** 3 / 12
        Iz = 1  # h * b ** 3 / 12
        J = 1  # Iy + Iz
        Ktc = E * S / L
        KT = G * J / L
        Kf1 = 12 * E * Iz / (L ** 3 * (1 + az))
        Kf2 = 12 * E * Iy / (L ** 3 * (1 + ay))
        Kf3 = -6 * E * Iy / (L ** 2 * (1 + ay))
        Kf4 = 6 * E * Iz / (L ** 2 * (1 + az))
        Kf5 = (4 + ay) * E * Iy / (L * (1 + ay))
        Kf6 = (4 + az) * E * Iz / (L * (1 + az))
        Kf7 = (2 - ay) * E * Iy / (L * (1 + ay))
        Kf8 = (2 - az) * E * Iz / (L * (1 + az))
        K_elem = np.array([[Ktc, 0, 0, 0, 0, 0, -Ktc, 0, 0, 0, 0, 0],  # 1
                           [0, Kf1, 0, 0, 0, Kf4, 0, -Kf1, 0, 0, 0, Kf4],
                           [0, 0, Kf2, 0, Kf3, 0, 0, 0, -Kf2, 0, Kf3, 0],
                           [0, 0, 0, KT, 0, 0, 0, 0, 0, -KT, 0, 0],
                           [0, 0, Kf3, 0, Kf5, 0, 0, 0, -Kf3, 0, Kf7, 0],
                           [0, Kf4, 0, 0, 0, Kf6, 0, -Kf4, 0, 0, 0, Kf8],
                           [-Ktc, 0, 0, 0, 0, 0, Ktc, 0, 0, 0, 0, 0],  # 7
                           [0, -Kf1, 0, 0, 0, -Kf4, 0, Kf1, 0, 0, 0, -Kf4],
                           [0, 0, -Kf2, 0, -Kf3, 0, 0, 0, Kf2, 0, -Kf3, 0],
                           [0, 0, 0, -KT, 0, 0, 0, 0, 0, KT, 0, 0],
                           [0, 0, Kf3, 0, Kf7, 0, 0, 0, -Kf3, 0, Kf5, 0],
                           [0, Kf4, 0, 0, 0, Kf8, 0, -Kf4, 0, 0, 0, Kf6]], dtype='float')
        return K_elem

    def base_change(self, P, M):
        """ method to perform base change from M matrix

        :param P: transfer matrix
        :param M: reference matrix
        :return:
        """
        return P @ M @ P.T

    def changement_coord(self):
        BB = []
        for i in range(len(self.mesh.element_list)):  # Une matrice de changement de coord par element
            # print("generation de la matrice de passage de l'element ", i + 1, ":")
            B = np.zeros([len(self.mesh.node_list) * 3, 6])
            noeud1 = self.mesh.element_list[i, 0]
            noeud2 = self.mesh.element_list[i, 1]
            B[(noeud1 - 1) * 3, 0] = 1
            B[(noeud1 - 1) * 3 + 1, 1] = 1
            B[(noeud1 - 1) * 3 + 2, 2] = 1
            B[(noeud2 - 1) * 3, 3] = 1
            B[(noeud2 - 1) * 3 + 1, 4] = 1
            B[(noeud2 - 1) * 3 + 2, 5] = 1
            BB.append(B)
        return BB

    def changement_coord_3D(self):
        BB = []
        for i in range(len(self.mesh.element_list)):  # Une matrice de changement de coord par element
            # print("generation de la matrice de passage de l'element ", i + 1, ":")
            B = np.zeros([len(self.mesh.node_list) * 6, 12])
            noeud1 = self.mesh.element_list[i, 0]
            noeud2 = self.mesh.element_list[i, 1]
            B[(noeud1 - 1) * 6, 0] = 1
            B[(noeud1 - 1) * 6 + 1, 1] = 1
            B[(noeud1 - 1) * 6 + 2, 2] = 1
            B[(noeud1 - 1) * 6 + 3, 3] = 1
            B[(noeud1 - 1) * 6 + 4, 4] = 1
            B[(noeud1 - 1) * 6 + 5, 5] = 1
            ###
            B[(noeud2 - 1) * 6, 0] = 1
            B[(noeud2 - 1) * 6 + 1, 1] = 1
            B[(noeud2 - 1) * 6 + 2, 2] = 1
            B[(noeud2 - 1) * 6 + 3, 3] = 1
            B[(noeud2 - 1) * 6 + 4, 4] = 1
            B[(noeud2 - 1) * 6 + 5, 5] = 1
            BB.append(B)
        return BB

    def get_length(self, element):
        noeud1 = element[0]
        noeud2 = element[1]
        if self.mesh.dim == 2:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            L_e = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
        elif self.mesh.dim == 3:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            z_1 = self.mesh.node_list[noeud1 - 1, 2]
            z_2 = self.mesh.node_list[noeud2 - 1, 2]
            L_e = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2 + (z_2 - z_1) ** 2)
        return L_e

    def get_angle(self, element):
        """ Return the cosinus and the sinus associated with the angle of the element
        in the global coordinate

        :return: tuple with cosinus and sinus
        :rtype: 2-uple
        """
        noeud1 = element[0]
        noeud2 = element[1]
        if self.mesh.dim == 2:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            L_e = self.get_length(element)
            c = (x_2 - x_1) / L_e
            s = (y_2 - y_1) / L_e
        elif self.mesh.dim == 3:
            x_1 = self.mesh.node_list[noeud1 - 1, 0]
            x_2 = self.mesh.node_list[noeud2 - 1, 0]
            y_1 = self.mesh.node_list[noeud1 - 1, 1]
            y_2 = self.mesh.node_list[noeud2 - 1, 1]
            z_1 = self.mesh.node_list[noeud1 - 1, 2]
            z_2 = self.mesh.node_list[noeud2 - 1, 2]
            L_e = self.get_length(element)
            c = (x_2 - x_1) / L_e
            s = (y_2 - y_1) / L_e
        return c, s

    def get_bc(self):
        """Return the boundary condition in a matrix format

        :return: matrix with 1 if the dof is blocked and 0 if the dof is free
        :rtype: np.array
        """
        BC = np.zeros(3 * len(self.mesh.node_list))
        for i in self.lbc:
            BC[i] = 1
        BC = BC.reshape((len(self.mesh.node_list), 3))
        return BC

    def assemble_2d_stiffness_matrix(self):
        """ Return the global stiffness matrix of the mesh

        :return: matrix of size(dll*3*nb_node,dll*3*nb_node)
        :rtype: np.array

        """
        BB = self.changement_coord()
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        M_global = np.zeros([len(NL) * 3, len(NL) * 3])

        for i in range(len(EL)):
            element = EL[i]
            L_e = self.get_length(element)
            c, s = self.get_angle(element)
            rot = self.get_2d_rotation_matrix(c, s)
            h, b = self.mesh.Section[i, 0], self.mesh.Section[i, 1]
            # rotation matrice elem
            K_rot = rot @ self.get_2d_element_stiffness_matrix(L_e, h, b) @ rot.T
            M_global = M_global + self.base_change(BB[i], K_rot)
            if self.mesh.debug == True:
                print("element " + str(i + 1) + " :")
                print(BB[i])
                print(rot)
                print("matrice elementaire : ")
                print(self.get_2d_element_stiffness_matrix(L_e, h, b))
                print(K_rot)
        return M_global

    def assemble_2d_mass_matrix(self):
        """Assembles the global stiffness using the sparse COO format.

        :returns: The global mass matrix
        :rtype: :class:`scipy.sparse.coo_matrix`
        """

        pass


    def assemblage_3D(self):
        """ Return the global stiffness matrix of the mesh

        :return: matrix of size(dll*3*nb_node,dll*3*nb_node)
        :rtype: np.array

        """
        BB = self.changement_coord_3D()
        NL = self.mesh.node_list
        EL = self.mesh.element_list
        M_global = np.zeros([len(NL) * 6, len(NL) * 6])
        for i in range(len(EL)):
            element = EL[i]
            L_e = self.get_length(element)
            rot = self.get_3d_rotation_matrix(NL[element[1] - 1])
            h, b = self.mesh.Section[i, 0], self.mesh.Section[i, 1]

            # rotation matrice elem
            K_rot = rot.dot(self.get_3d_element_stiffness_matrix(L_e, h, b)).dot(np.transpose(rot))
            M_global = M_global + self.base_change(BB[i], K_rot)
            if self.mesh.debug == True:
                print("element " + str(i + 1) + " :")
                print(BB[i])
                print(rot)
                print("matrice elementaire : ")
                print(self.get_3d_element_stiffness_matrix(L_e, h, b))
                print(K_rot)
        return M_global

    def solver_frame(self):
        """ Solves Ku=f_ext for *u* using the direct method.

        :return:
        """
        self.bc = np.delete(self.bc, self.lbc, axis=1)


        if self.mesh.dim == 2:
            K_glob = self.assemble_2d_stiffness_matrix()

        elif self.mesh.dim == 3:
            K_glob = self.assemblage_3D()

        K_glob_r = self.bc.T @ K_glob @ self.bc
        F = np.vstack(self.load.flatten())
        F_r = self.bc.T @ F
        U_r = np.linalg.inv(K_glob_r) @ F_r

        self.U = self.bc @ U_r
        self.React = K_glob @ self.U - F
        #self.S = self.stress()

    def get_local_U(self, element):
        """Retourne le vecteur deplacement dans le repère local à partir du vecteur dans le repère global"""

        i, j = element[0] - 1, element[1] - 1
        if self.mesh.dim == 2:
            c, s = self.get_angle(element)
            rot = self.get_2d_rotation_matrix(c, s)
            global_X = np.concatenate((self.U[i * 3:i * 3 + 3], self.U[j * 3:j * 3 + 3]), axis=None)
        elif self.mesh.dim == 3:
            rot = self.get_3d_rotation_matrix(self.mesh.node_list[j])
            global_X = np.concatenate((self.U[i * 6:i * 6 + 6], self.U[j * 6:j * 6 + 6]), axis=None)
        local_X = rot.T @ global_X
        return local_X

    def get_internal_force(self):

        internal_forces = []
        EL = self.mesh.element_list
        NoE = len(EL)

        for el in EL:
            L_e = self.get_length(el)
            int_f = self.get_2d_element_stiffness_matrix(L_e, 1, 1) @ self.get_local_U(el)
            internal_forces.append(int_f)

        internal_forces = np.array(internal_forces)

        if self.mesh.dim == 2:
            internal_forces = np.reshape(internal_forces, (NoE * 2, 3))
        else :
            internal_forces = np.reshape(internal_forces, (NoE * 2, 6))

        return internal_forces


    def get_local_F(self, element):
        """Retourne le vecteur force dans le repère local à partir du vecteur dans le repère global"""

        i, j = element[0] - 1, element[1] - 1
        if self.mesh.dim == 2:
            c, s = self.get_angle(element)
            rot = self.get_2d_rotation_matrix(c, s)
            global_X = np.concatenate((self.U[i * 3:i * 3 + 3], self.U[j * 3:j * 3 + 3]), axis=None)
            local_X = rot.T @ global_X
            local_U = self.get_local_U(element)
            L_e = self.get_length(element)
            h, b = 1, 1  # self.mesh.Section[i, 0], self.mesh.Section[i, 1]
            # rotation matrice elem
            k = self.get_2d_element_stiffness_matrix(L_e, h, b)
            load_element = np.concatenate((self.load[i], self.load[j]), axis=None)
            local_f = k @ local_U
            print(local_f)
        return local_f

    def calcul_stresses(self, elem):
        # TODO : bien prendre les valeurs dans le repère local de l'element
        # TODO : récupérer les dimensions de l'element
        """calcul les différentes contraintes sur un elmeent donné"""
        NL = self.mesh.node_list
        node_i, node_j = elem[0] - 1, elem[1] - 1
        L = self.get_length(elem)
        U = self.U
        U = self.get_local_U(elem)
        G = self.E / 2 / (1 + 0.3)
        h, b = 4, 2  # self.mesh.Section[i,0], self.mesh.Section[i,1]
        Iy = b * h ** 3 / 12
        Iz = h * b ** 3 / 12
        k = 5 / 6

        if self.mesh.dim == 2:
            epsilon_x = (U[3] - U[0]) / L
            sigma_x = self.E * epsilon_x  # / 1E6
            sigma_fy = self.E * h * (U[5] - U[2]) / L  # / 1E6
            tau_y = 0 / 1E6
            sigma_VM = math.sqrt((sigma_x + sigma_fy) ** 2 + 3 * (tau_y) ** 2)
            sigma_T = math.sqrt((sigma_x + sigma_fy) ** 2 + 4 * (tau_y) ** 2) / 1E6
            if self.mesh.debug == True:
                print("déformation (en mm) =", epsilon_x * 1E3)
                print("contrainte normale (en MPa) =", sigma_x)
                print("contrainte normale de flexion (en MPa) =", sigma_fy)
                print("contrainte cisaillement de flexion (en MPa) =", tau_y)
                print("contrainte Von Mises (en MPa) =", sigma_VM)
                print("contrainte Tresca (en MPa) =", sigma_T)
            return np.array([[sigma_x, sigma_fy, tau_y, sigma_VM, sigma_T]])

        elif self.mesh.dim == 3:
            RR = self.get_3d_rotation_matrix(NL[elem[1] - 1])
            rot_max = RR[0:6, 0:6]
            Ui = np.transpose(rot_max).dot(np.array(U[6 * node_i: 6 * node_i + 6]))
            Uj = np.transpose(rot_max).dot(np.array(U[6 * node_j: 6 * node_j + 6]))
            epsilon_x = (Uj[0] - Ui[0]) / L
            sigma_x = self.E * epsilon_x
            tau_x = G * (Uj[1] - Ui[1]) / L * max(h, b)
            sigma_fy = self.E * h * (U[6 * node_j + 5] - U[6 * node_i + 5]) / L
            sigma_fz = self.E * b * (U[6 * node_j + 4] - U[6 * node_i + 4]) / L
            Ay = 12 * self.E * Iy / (k * G * h * b * L ** 2 + 12 * self.E * Iy)
            Az = 12 * self.E * Iz / (k * G * h * b * L ** 2 + 12 * self.E * Iz)
            tau_y = -G * Ay * (2 * U[6 * node_i + 1] + U[6 * node_i + 5] * L - 2 * U[6 * node_j + 1] + U[
                6 * node_j + 5] * L) / L ** 2
            tau_z = -G * Az * (2 * U[6 * node_i + 2] + U[6 * node_i + 4] * L - 2 * U[6 * node_j + 2] + U[
                6 * node_j + 4] * L) / L ** 2
            sigma_VM = np.sqrt((sigma_x + sigma_fy + sigma_fz) ** 2 + 3 * (tau_x + tau_y + tau_z) ** 2)
            if self.mesh.debug == True:
                print("contrainte normale (en MPa) =", sigma_x[0] / 1E6)
                print("contrainte normale de flexion (en MPa) =", sigma_fz[0] / 1E6)
                print("contrainte normale de flexion (en MPa) =", sigma_fy[0] / 1E6)
                print("contrainte cisaillement de torsion (en MPa) =", tau_x[0] / 1E6)
                print("contrainte cisaillement de flexion (en MPa) =", tau_y[0] / 1E6)
                print("contrainte cisaillement de flexion (en MPa) =", tau_z[0] / 1E6)
                print("contrainte Von Mises (en MPa) =", sigma_VM[0] / 1E6)
            return np.array([[sigma_x, sigma_fy, sigma_fz, tau_x, tau_y, tau_z, sigma_VM]])

    def stress(self):
        EL = self.mesh.element_list
        for elem in EL:
            self.S = np.append(self.S, self.calcul_stresses(elem), axis=0)
        return self.S

    def get_res(self):
        # local vector
        F_local = np.empty((0, len(self.mesh.node_list) * 3))
        U_local = np.empty((0, len(self.mesh.node_list) * 3))


        for el in self.mesh.element_list:
            fl = self.get_local_F(el)
            ul = self.get_local_U(el)
            F_local = np.concatenate((F_local, [fl]), axis=None)
            U_local = np.concatenate((U_local, [ul]), axis=None)


        self.res = {}
        self.res['U'] = self.U
        self.res['u'] = U_local
        self.res['React'] = self.React
        self.res['F'] = self.load
        self.res['f'] = F_local.reshape((len(self.mesh.element_list)*2,3))
        #self.res['stress'] = self.S
        self.res['node'] = self.mesh.node_list
        self.res['element'] = self.mesh.element_list
        return self.res

    # -----------------
    # display functions
    # -----------------

    # TODO: Faire un script dédié pour l'affichage

    def U_table(self):
        tab = pt()
        if self.mesh.dim == 2:
            tab.field_names = ["Node", "Ux (m)", "Uy (m)", "Phi (rad)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.U[i * 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 3 + 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 3 + 2], precision=2, exp_digits=2)])
        else:
            tab.field_names = ["Node", "Ux (m)", "Uy (m)", "Uz (m)", "Phix (rad)", "Phiy (rad)", "Phiz (rad)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.U[i * 6], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 2], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 4], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 5], precision=2, exp_digits=2)])
        print(tab)

    def R_table(self):
        tab = pt()
        if self.mesh.dim == 2:
            tab.field_names = ["Node", "Fx (N)", "Fy (N)", "Mz (N.m)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.React[i * 3][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 3 + 1][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 3 + 2][0], precision=2, exp_digits=2)])
        else:
            tab.field_names = ["Node", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.React[i * 6][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 1][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 2][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 3][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 4][0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.React[i * 6 + 5][0], precision=2, exp_digits=2)])
        print(tab)

    def S_table(self):
        tab = pt()
        if self.mesh.dim == 2:
            tab.field_names = ["Elem", "Sx (MPa)", "Sf (MPa)", "Ty (MPa)", "SVM (MPa)", "Tresca (MPa)"]
            for i in range(len(self.mesh.element_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.S[i, 0], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i, 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i, 2], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i, 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.S[i, 4], precision=2, exp_digits=2)])
        else:
            tab.field_names = ["Node", "Ux (m)", "Uy (m)", "Uz (m)", "Phix (rad)", "Phiy (rad)", "Phiz (rad)"]
            for i in range(len(self.mesh.node_list)):
                tab.add_row([int(i + 1),
                             np.format_float_scientific(self.U[i * 6], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 1], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 2], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 3], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 4], precision=2, exp_digits=2),
                             np.format_float_scientific(self.U[i * 6 + 5], precision=2, exp_digits=2)])
        print(tab)

    # --------------
    # plot functions
    # --------------

    def plot_model(self, ax=None, supports=True, loads=True, deformed=False, def_scale=1):
        """Method used to plot the structural mesh in the undeformed and/or deformed state. If no
        axes object is provided, a new axes object is created. N.B. this method is adapted from the
        MATLAB code by F.P. van der Meer: plotGeom.m.

        :param ax: Axes object on which to plot
        :type ax: :class:`matplotlib.axes.Axes`
        :param bool supports: Whether or not the freedom case supports are rendered
        :param bool loads: Whether or not the load case loads are rendered
        :param bool undeformed: Whether or not the undeformed structure is plotted
        :param bool deformed: Whether or not the deformed structure is plotted
        :param float def_scale: Deformation scale used for plotting the deformed structure
        :param bool dashed: Whether or not to plot the structure with dashed lines if only the undeformed structure is to be plotted
        """

        if ax is None:
            (fig, ax) = plt.subplots()

        if not deformed:
            self.mesh.plot_mesh(ax=ax)
        else:
            pass

        # set initial plot limits
        (xmin, xmax, ymin, ymax, _, _) = self.mesh.get_node_lims()
        #ax.set_xlim(xmin - 1e-12, xmax + 1e-12)
        #ax.set_ylim(ymin - 1e-12, ymax + 1e-12)

        # get 2% of the maxmimum dimension
        small = 0.02 * max(xmax - xmin, ymax - ymin)

        if loads:
            print("================ Plot loads ====================")
            # iterate on the loaded nodes
            for i in self.get_nodes_loaded():
                max_force = np.abs(self.load).max()
                print("traitement du node %s" %i)
                print("max force : %s" %max_force)
                print(self.load[i])

                for j in range(len(self.load[i])):
                    load = self.load[i,j]
                    # get the dof of the apply load
                    if not load == 0:
                        dof = j
                        print("dof :", dof)
                        self.plot_nodal_load(ax, i, load, max_force, dof, small)

        if supports:
            print("================ Plot supports ====================")
            BC = self.get_bc()
            for i, bc in enumerate(BC):
                print("traitement node %s" % i)
                print("bc :", bc)
                self.plot_supports(ax, list(bc), i+1, small)

        # plot layout
        plt.axis('tight')
        ax.set_xlim(self.wide_lim(ax.get_xlim()))
        ax.set_ylim(self.wide_lim(ax.get_ylim()))

        limratio = np.diff(ax.get_ylim())/np.diff(ax.get_xlim())

        if limratio < 0.5:
            ymid = np.mean(ax.get_ylim())
            ax.set_ylim(ymid + (ax.get_ylim() - ymid) * 0.5 / limratio)
        elif limratio > 1:
            xmid = np.mean(ax.get_xlim())
            ax.set_xlim(xmid + (ax.get_xlim() - xmid) * limratio)

        ax.set_aspect(1)
        plt.box(on=None)
        plt.show()

        return

    def plot_nodal_load(self, ax, node_index, val, max_force, dof, small):
        """Plots a graphical representation of a nodal force. A straight arrow is plotted for a
        translational load and a curved arrow is plotted for a moment.

        :param ax: Axes object on which to plot
        :type ax: :class:`matplotlib.axes.Axes`
        :param float max_force: Maximum translational nodal load
        :param float small: A dimension used to scale the support
        :param get_support_angle: A function that returns the support angle and the number of
            connected elements
        :type get_support_angle: :func:`feastruct.post.post.PostProcessor.get_support_angle`
        :param analysis_case: Analysis case
        :type analysis_case: :class:`~feastruct.fea.cases.AnalysisCase`
        :param bool deformed: Represents whether or not the node locations are deformed based on
            the results of case id
        :param float def_scale: Value used to scale deformations
        """

        node = self.mesh.node_list[node_index]
        x = node[0]
        y = node[1]

        print(x,y)

        if max_force == 0:
            val = 1
        else:
            val = val / max_force

        offset = 0.5 * small
        (angle, num_el) = self.get_support_angle(node_index+1)
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

        ax.plot(rl[0, s:e] + x, rl[1, s:e] + y, 'k-', linewidth=2)
        ax.add_patch(Polygon(np.transpose(rp), facecolor='k'))
        print("plot force")

        return

    def plot_dist_load(self, ax):
        pass

    def plot_supports(self, ax, bc, node_index, small):
        if bc not in ([1, 1, 0], [0, 1, 0]):
            (angle, num_el) = self.get_support_angle(node_index)
            print(angle, num_el)

        if bc == [1, 0, 0]:
            # ploy a y-roller
            angle = round(angle / 180) * 180
            self.plot_xysupport(ax, angle, True, num_el == 1, small, analysis_case, deformed,
                                def_scale)

        elif bc == [0, 1, 0]:
            (angle, num_el) = self.get_support_angle(node_index, 1)
            # plot an x-roller
            if np.mod(angle + 1, 180) < 2:  # prefer support below
                angle = 90
            else:
                angle = round((angle + 90) / 180) * 180 - 90

            node = self.mesh.node_list[node_index - 1]
            self.plot_xysupport(ax, node, angle, True, num_el == 1, small)
            return

        elif bc == [1, 1, 0]:
            # plot a hinge
            (angle, num_el) = self.get_support_angle(node_index, 1)

            node = self.mesh.node_list[node_index-1]
            self.plot_xysupport(ax, node, angle, False, num_el == 1, small)
            return

        elif bc == [0, 0, 1]:
            ax.plot(self.node.x, self.node.y, 'kx', markersize=8)

        else:
            # plot a support with moment bc
            if bc == [1, 1, 1]:
                # plot a fixed support
                s = np.sin(angle * np.pi / 180)
                c = np.cos(angle * np.pi / 180)
                rot_mat = np.array([[c, -s], [s, c]])
                line = np.array([[0, 0], [-1, 1]]) * small
                rect = np.array([[-0.6, -0.6, 0, 0], [-1, 1, 1, -1]]) * small
                ec = 'none'

            elif bc == [1, 0, 1]:
                # plot y-roller block
                angle = round(angle / 180) * 180
                s = np.sin(angle * np.pi / 180)
                c = np.cos(angle * np.pi / 180)
                rot_mat = np.array([[c, -s], [s, c]])
                line = np.array([[-0.85, -0.85], [-1, 1]]) * small
                rect = np.array([[-0.6, -0.6, 0, 0], [-1, 1, 1, -1]]) * small
                ec = 'k'

            elif bc == [0, 1, 1]:
                # plot x-roller block
                angle = round((angle + 90) / 180) * 180 - 90
                s = np.sin(angle * np.pi / 180)
                c = np.cos(angle * np.pi / 180)
                rot_mat = np.array([[c, -s], [s, c]])
                line = np.array([[-0.85, -0.85], [-1, 1]]) * small
                rect = np.array([[-0.6, -0.6, 0, 0], [-1, 1, 1, -1]]) * small
                ec = 'k'
            else:
                return

            rot_line = np.matmul(rot_mat, line)
            rot_rect = np.matmul(rot_mat, rect)

            rot_line[0, :] += self.mesh.node_list[node_index-1,0]
            rot_line[1, :] += self.mesh.node_list[node_index-1,1]
            rot_rect[0, :] += self.mesh.node_list[node_index-1,0]
            rot_rect[1, :] += self.mesh.node_list[node_index-1,1]

        ax.plot(rot_line[0, :], rot_line[1, :], 'k-', linewidth=1)
        ax.add_patch(Polygon(np.transpose(rot_rect), facecolor=(0.7, 0.7, 0.7), edgecolor=ec))

    def plot_xysupport(self, ax, node, angle, roller, hinge, small):

        # determine coordinates of node
        x = node[0]
        y = node[1]

        # determine coordinates of triangle
        dx = small
        h = np.sqrt(3) / 2
        triangle = np.array([[-h, -h, -h, 0, -h], [-1, 1, 0.5, 0, -0.5]]) * dx
        s = np.sin(angle * np.pi / 180)
        c = np.cos(angle * np.pi / 180)
        rot_mat = np.array([[c, -s], [s, c]])
        rot_triangle = np.matmul(rot_mat, triangle)

        if roller:
            line = np.array([[-1.1, -1.1], [-1, 1]]) * dx
            rot_line = np.matmul(rot_mat, line)
            ax.plot(rot_line[0, :] + x, rot_line[1, :] + y, 'k-', linewidth=1)
        else:
            rect = np.array([[-1.4, -1.4, -h, -h], [-1, 1, 1, -1]]) * dx
            rot_rect = np.matmul(rot_mat, rect)
            rot_rect[0, :] += x
            rot_rect[1, :] += y
            ax.add_patch(Polygon(np.transpose(rot_rect), facecolor=(0.7, 0.7, 0.7)))

        ax.plot(rot_triangle[0, :] + x, rot_triangle[1, :] + y, 'k-', linewidth=1)

        if hinge:
            ax.plot(x, y, 'ko', markerfacecolor='w', linewidth=1, markersize=4)

    def plot_force(self, ax):
        pass

    def plot_reactions(self, ax):
        """Method used to generate a plot of the reaction forces.

        :param ax: Axes object on which to plot
        :type ax: :class:`matplotlib.axes.Axes`
        :return:
        """

        (fig, ax) = plt.subplots()

        # get size of structure
        (xmin, xmax, ymin, ymax, _, _) = self.mesh.get_node_lims()

        # determine maximum reaction force
        max_reaction = 0

        for line in self.get_bc():
            if line in [0, 0, 1]:
                reaction = support.get_reaction()
                max_reaction = max(max_reaction, abs(reaction))

        small = 0.02 * max(xmax - xmin, ymax - ymin)

        # plot reactions
        for support in analysis_case.freedom_case.items:
            support.plot_reaction(
                ax=ax, max_reaction=max_reaction, small=small,
                get_support_angle=self.get_support_angle, analysis_case=analysis_case)

        # plot the undeformed structure
        self.plot_geom(analysis_case=analysis_case, ax=ax, supports=False)

    def get_support_angle(self, node, prefer_dir=None):
        """Given a node object, returns the optimal angle to plot a support. Essentially finds the
        average angle of the connected elements and considers a preferred plotting direction. N.B.
        this method is adapted from the MATLAB code by F.P. van der Meer: plotGeom.m.

        :param node: Node object
        :type node: :class:`~feastruct.fea.node.node`
        :param int prefer_dir: Preferred direction to plot the support, where 0 corresponds to the
            x-axis and 1 corresponds to the y-axis
        """

        print("===> Enter get support angle")
        print(node)
        # find angles to connected elements
        phi = []
        num_el = 0

        # loop through each element in the mesh
        for el in self.mesh.element_list:
            print(el)
            # if the current element is connected to the node
            if node in el:
                num_el += 1
                # loop through all the nodes connected to the element
                for el_node in el:
                    print(el_node, node)
                    # if the node is not the node in question
                    if not el_node == node:
                        node_test = self.mesh.node_list[node-1]
                        node_connected = self.mesh.node_list[el_node-1]
                        dx = [node_connected[0] - node_test[0], node_connected[1] - node_test[1]]
                        phi.append(np.arctan2(dx[1], dx[0]) / np.pi * 180)

        phi.sort()
        phi.append(phi[0] + 360)
        i0 = np.argmax(np.diff(phi))
        angle = (phi[i0] + phi[i0+1]) / 2 + 180

        print(phi)

        if prefer_dir is not None:
            if prefer_dir == 1:
                if max(np.sin([phi[i0] * np.pi / 180, phi[i0+1] * np.pi / 180])) > -0.1:
                    angle = 90
            elif prefer_dir == 0:
                if max(np.cos([phi[i0] * np.pi / 180, phi[i0+1] * np.pi / 180])) > -0.1:
                    angle = 0

        return (angle, num_el)

    def wide_lim(self, x):
        """Returns a tuple corresponding to the axis limits (x) stretched by 2% on either side.

        :param x: List containing axis limits e.g. [xmin, xmax]
        :type x: list[float, float]
        :returns: Stretched axis limits (x1, x2)
        :rtype: tuple(float, float)
        """

        x2 = max(x)
        x1 = min(x)
        dx = x2-x1
        f = 0.02

        return (x1 - f * dx, x2 + f * dx)

def test_1():
    mesh = Mesh(2, debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([0, 10])  # inches
    mesh.add_node([10, 10])  # inches
    mesh.add_element([1, 2], "barre", "b", 15, 15)
    mesh.add_element([2, 3], "barre", "r", 15, 15)
    mesh.add_element([1, 3], "barre", "r", 15, 15)
    #mesh.plot_mesh()
    print(mesh.node_list)

    f = FEM_Model(mesh)
    f.apply_load([500, -1000, 0], 2)
    f.apply_load([500, -1000, 0], 3)
    print("load", f.load)
    f.get_nodes_loaded()

    f.apply_bc([1, 1, 0], 1)
    f.apply_bc([1, 1, 1], 3)
    print(f.lbc)
    print("BC matrix :", f.get_bc())
    f.plot_model()
    f.solver_frame()

def test_2():
    mesh = Mesh(2, debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([10, 0])  # inches
    mesh.add_node([20, 10])  # inches
    mesh.add_element([1, 2], "barre", "b", 15, 15)
    mesh.add_element([2, 3], "barre", "r", 15, 15)
    # mesh.plot_mesh()
    print(mesh.node_list)

    f = FEM_Model(mesh)
    f.apply_load([0, -1000, 0], 2)
    print("load", f.load)
    f.get_nodes_loaded()

    f.apply_bc([0, 1, 1], 1)
    f.apply_bc([1, 1, 0], 3)
    print(f.lbc)
    print("BC matrix :", f.get_bc())
    f.plot_model()
    f.solver_frame()

def test_3():
    mesh = Mesh(2, debug=False)
    mesh.add_node([0, 0])
    mesh.add_node([0, 10])  # inches
    mesh.add_node([10, 10])  # inches
    mesh.add_node([10, 0])  # inches
    mesh.add_element([1, 2], "barre", "b", 15, 15)
    mesh.add_element([2, 3], "barre", "r", 15, 15)
    mesh.add_element([3, 4], "barre", "g", 15, 15)
    # mesh.plot_mesh()
    print(mesh.node_list)

    f = FEM_Model(mesh)
    f.apply_load([0, -1000, 0], 2)
    print("load", f.load)
    f.get_nodes_loaded()

    f.apply_bc([1, 1, 1], 1)
    f.apply_bc([1, 1, 1], 4)
    print(f.lbc)
    print("BC matrix :", f.get_bc())
    #f.plot_model()
    f.solver_frame()
    f.U_table()
    intf = f.get_internal_force()
    print(intf)

if __name__ == "__main__":
    test_3()

'''
TODO : 
    [x] arrondi en notation scientifique en python
    [x] visuel charge répartie
    [] bien gérer la génération d'une charge répartie et d'une charge ponctuelle
    [] sortie format json ou dictionnaire ?
    [] nettoyage du code 
    [] ajouter des docstrings
'''

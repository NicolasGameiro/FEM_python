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

    def apply_load(self, node_load, node):
        if node > len(self.mesh.node_list):
            print("Error : node specified not in the mesh")
        elif (len(node_load) == 3) or (len(node_load) == 6):
            self.load[node - 1, :] = node_load
            # print("nodal load applied")
            if self.mesh.debug == True:
                print(self.load)
        else:
            print("Error : uncorrect load format")

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
            Q = np.array([0, 0, -q * L / 2, -q * L ** 2 / 12, 0, 0,
                          0, 0, -q * L / 2, q * L ** 2 / 12, 0, 0])
            self.load[element[0] - 1] = self.load[element[0] - 1] + Q[:6]
            self.load[element[1] - 1] = self.load[element[1] - 1] + Q[6:12]
        self.dist_load = np.append(self.dist_load, [[element[0], element[1], q]], axis=0)
        # print(self.dist_load)

    def apply_bc(self, node_bc, node):
        if node > len(self.mesh.node_list):
            print("Error : node specified not in the mesh")
        elif len(node_bc) == 3:
            for i in range(len(node_bc)):
                if node_bc[i] == 1:
                    self.lbc.append(i + 3 * (node - 1))
            # print("boundary condition applied")
        elif len(node_bc) == 6:
            for i in range(len(node_bc)):
                if node_bc[i] == 1:
                    self.lbc.append(i + 6 * (node - 1))
            # print("boundary condition applied")
        else:
            print("Error : uncorrect bc format")

    def Rot(self, c, s):
        """ Rotation matrix in 2D
        """
        Rotation_matrix = np.array([[c, -s, 0, 0, 0, 0],
                                    [s, c, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, c, -s, 0],
                                    [0, 0, 0, s, c, 0],
                                    [0, 0, 0, 0, 0, 1]])
        return Rotation_matrix

    def Rot_3D(self, vec2):
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

    def K_elem(self, L_e, h, b):
        S = h * b  # * 1e-4
        I = b * h ** 3 / 12  # * 1e-8
        K_elem = self.E / L_e * np.array([[S, 0, 0, -S, 0, 0],
                                          [0, 12 * I / L_e ** 2, 6 * I / L_e, 0, -12 * I / L_e ** 2, 6 * I / L_e],
                                          [0, 6 * I / L_e, 4 * I, 0, -6 * I / L_e, 2 * I],
                                          [-S, 0, 0, S, 0, 0],
                                          [0, -12 * I / L_e ** 2, -6 * I / L_e, 0, 12 * I / L_e ** 2, -6 * I / L_e],
                                          [0, 6 * I / L_e, 2 * I, 0, -6 * I / L_e, 4 * I]])
        return K_elem

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

    def K_elem_3d(self, L: float, h: float, b: float, E: float = 1, nu: float = 0.3, ay: float = 0,
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

    def changement_base(self, P, M):
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

    def assemblage_2D(self):
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
            rot = self.Rot(c, s)
            h, b = self.mesh.Section[i, 0], self.mesh.Section[i, 1]
            # rotation matrice elem
            K_rot = rot @ self.K_elem(L_e, h, b) @ rot.T
            M_global = M_global + self.changement_base(BB[i], K_rot)
            if self.mesh.debug == True:
                print("element " + str(i + 1) + " :")
                print(BB[i])
                print(rot)
                print("matrice elementaire : ")
                print(self.K_elem(L_e, h, b))
                print(K_rot)
        return M_global

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
            rot = self.Rot_3D(NL[element[1] - 1])
            h, b = self.mesh.Section[i, 0], self.mesh.Section[i, 1]

            # rotation matrice elem
            K_rot = rot.dot(self.K_elem_3d(L_e, h, b)).dot(np.transpose(rot))
            M_global = M_global + self.changement_base(BB[i], K_rot)
            if self.mesh.debug == True:
                print("element " + str(i + 1) + " :")
                print(BB[i])
                print(rot)
                print("matrice elementaire : ")
                print(self.K_elem_3d(L_e, h, b))
                print(K_rot)
        return M_global

    def solver_frame(self):
        self.bc = np.delete(self.bc, self.lbc, axis=1)
        if self.mesh.dim == 2:
            K_glob = self.assemblage_2D()
        elif self.mesh.dim == 3:
            K_glob = self.assemblage_3D()
        K_glob_r = self.bc.T @ K_glob @ self.bc
        F = np.vstack(self.load.flatten())
        F_r = self.bc.T @ F
        U_r = np.linalg.inv(K_glob_r) @ F_r
        self.U = self.bc @ U_r
        self.React = K_glob @ self.U - F
        self.S = self.stress()

    def get_local_U(self, element):
        """Retourne le vecteur deplacement dans le repère local à partir du vecteur dans le repère global"""

        i, j = element[0] - 1, element[1] - 1
        if self.mesh.dim == 2:
            c, s = self.get_angle(element)
            rot = self.Rot(c, s)
            global_X = np.concatenate((self.U[i * 3:i * 3 + 3], self.U[j * 3:j * 3 + 3]), axis=None)
        elif self.mesh.dim == 3:
            rot = self.Rot_3D(self.mesh.node_list[j])
            global_X = np.concatenate((self.U[i * 6:i * 6 + 6], self.U[j * 6:j * 6 + 6]), axis=None)
        local_X = rot.T @ global_X
        return local_X

    def get_local_F(self, element):
        """Retourne le vecteur force dans le repère local à partir du vecteur dans le repère global"""

        i, j = element[0] - 1, element[1] - 1
        if self.mesh.dim == 2:
            c, s = self.get_angle(element)
            rot = self.Rot(c, s)
            global_X = np.concatenate((self.U[i * 3:i * 3 + 3], self.U[j * 3:j * 3 + 3]), axis=None)
            local_X = np.transpose(rot).dot(global_X)
            local_U = self.get_local_U(element)
            L_e = self.get_length(element)
            h, b = 1, 1  # self.mesh.Section[i, 0], self.mesh.Section[i, 1]
            # rotation matrice elem
            k = self.K_elem(L_e, h, b)
            local_f = k @ local_U
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
            RR = self.Rot_3D(NL[elem[1] - 1])
            rot_max = RR[0:6, 0:6]
            Ui = np.transpose(rot_max).dot(np.array(U[6 * node_i: 6 * node_i + 6]))
            Uj = np.transpose(rot_max).dot(np.array(U[6 * node_j: 6 * node_j + 6]))
            epsilon_x = (Uj[6] - Ui[0]) / L
            sigma_x = self.E * epsilon_x
            tau_x = G * (Uj[7] - Ui[1]) / L * max(h, b)
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
        self.res['f'] = F_local
        self.res['stress'] = self.S
        self.res['node'] = self.mesh.node_list
        self.res['element'] = self.mesh.element_list
        return self.res

    ### Display tables
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


if __name__ == "__main__":
    test_3d()

'''
TODO : 
    [x] arrondi en notation scientifique en python
    [x] visuel charge répartie
    [] bien gérer la génération d'une charge répartie et d'une charge ponctuelle
    [] sortie format json ou dictionnaire ?
    [] nettoyage du code 
    [] ajouter des docstrings
'''

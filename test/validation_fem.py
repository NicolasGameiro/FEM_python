# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 23:33:58 2022

@author: ngameiro
"""

from Code_FEM_v5 import *
import numpy as np
import matplotlib.pyplot as plt

def validation_cas1(nb_elem = 2) : 
    """ Cas 1 : poutre encastrée-libre effort ponctuel à son extrémité
    
    :return: DESCRIPTION
    :rtype: TYPE
    
    """
    ### Analytique 
    F = -1000 #N
    L = 1 #m
    E = 210E9 #Pa
    h, b = 10, 10
    I = h*b**3/12 * 1E-8 #m4
    x = np.linspace(0,L,20)
    d = lambda x : F*x**2*(3*L-x)/(6*E*I)
    w = d(x)
    
    ### FEM
    mesh = Mesh(2,[],[],debug = False)
    for i in range(nb_elem+1):
        mesh.add_node([i*L/(nb_elem),0])
    for i in range(nb_elem):
        mesh.add_element([i+1,i+2], "barre", "b", h, b)
    #mesh.geom()
    f = FEM_Model(mesh, E = E)
    f.apply_load([0,F,0],nb_elem+1)
    f.apply_bc([1,1,1],1)
    f.solver_frame()
    U, R, res = f.get_res()
    ### Deformation
    Uy = [U[i] for i in range(1,len(U),3)]
    X = np.linspace(0,L,len(mesh.node_list))
    erreur = [abs(d(X[i]) - Uy[i][0]) for i in range(len(X))]
    err_max = max(erreur)
    print('f_max (analytique) = ', np.format_float_scientific(max(abs(w)), precision = 2, exp_digits=2))
    print('f_max (FEM) = ', np.format_float_scientific(max(Uy, key=abs), precision = 2, exp_digits=2))
    print("erreur max :", err_max)
    ### Contrainte 
    S = f.stress()
    sf = S[1]
    M_max = F*L
    sigma_f_a = M_max/I*h/2*1E-2
    print("contrainte de flexion analytique (MPa) =", np.format_float_scientific(sigma_f_a/1E6, precision = 2, exp_digits=2))
    print("contrainte de flexion FEM (MPa) = ", np.format_float_scientific(sf, precision = 2, exp_digits=2))
    ecart_abs = np.round(abs(sigma_f_a/1E6 - sf),2)
    ecart_relatif = np.round(abs(sigma_f_a - sf)/abs(sigma_f_a)*100, 2)
    print("ecart absolu =", ecart_abs, " MPa (ecart_relatif = ", ecart_relatif, "%)")
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x, w, color='blue', label="E-B")
    ax.plot(X,Uy, 'o-', color='red', label="FEM")
    ax.legend()
    
    
    x = np.arange(len(X)) # the label locations
    labels = str(x)
    width = 0.35 # the width of the bars
    
    ax = fig.add_subplot(1, 2, 2)
    rects1 = ax.bar(x - width/2, erreur, width, label='Erreur')
    #ax.bar_label(rects1, padding=3)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Erreur')
    ax.set_title('Erreur')
    #ax.set_xticks(x, labels)
    ax.legend()
    
    return err_max

def validation_cas2(nb_elem = 1) : 
    """ Cas 2 : poutre encastrée-libre charge répartie
    
    :return: DESCRIPTION
    :rtype: TYPE
    
    """
    ### Analytique 
    q = 10
    L = 10
    E = 10E9
    I = 0.1*0.1**3/12
    x = np.linspace(0,L,20)
    d = lambda a : -q*a**2*(6*L**2-4*L*a+a**2)/(24*E*I)
    w = d(x)
    
    ### FEM
    mesh = Mesh(2,[],[],debug = False)
    for i in range(nb_elem+1):
        mesh.add_node([i*L/(nb_elem),0])
    for i in range(nb_elem):
        mesh.add_element([i+1,i+2], "barre", "b", 10, 10)
    #mesh.geom()
    f = FEM_Model(mesh, E = E)
    for i in range(nb_elem):
        f.apply_distributed_load(q, mesh.element_list[i])
    f.apply_bc([1,1,1],1)
    f.solver_frame()
    U, R, res = f.get_res()
    Uy = [U[i] for i in range(1,len(U),3)]
    X = np.linspace(0,L,len(mesh.node_list))
    erreur = [abs(d(X[i]) - Uy[i][0]) for i in range(len(X))]
    err_max = max(erreur)
    print("erreur max :", err_max)
    
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x, w, color='blue', label="E-B")
    ax.plot(X,Uy, 'o-', color='red', label="FEM")
    ax.legend()
    
    x = np.arange(len(X)) # the label locations
    labels = str(x)
    width = 0.35 # the width of the bars
    
    ax = fig.add_subplot(1, 2, 2)
    rects1 = ax.bar(x - width/2, erreur, width, label='Erreur')
    #ax.bar_label(rects1, padding=3)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Erreur')
    ax.set_title('Erreur')
    #ax.set_xticks(x, labels)
    ax.legend()
    """
    return err_max

def validation_cas3(nb_elem = 1) : 
    """ Cas 3 : portique avec force extrémité
    
    :return: DESCRIPTION
    :rtype: TYPE
    
    """
    ### Analytique 
    F = -1000
    L = 1
    E = 210E9
    I = 0.1*0.1**3/12
    ux = 0.3E-3
    uy = -0.8E-3
    
    ### FEM
    mesh = Mesh(2,[],[],debug = False)
    mesh.add_node([0,0])
    mesh.add_node([0,L])
    mesh.add_node([L,L])
    mesh.add_element([1,2], "barre", "b", 10, 10)
    mesh.add_element([2,3], "barre", "b", 10, 10)
    mesh.plot_mesh()
    f = FEM_Model(mesh, E = E)
    f.apply_load([0,F,0],3)
    f.apply_bc([1,1,1],1)
    f.solver_frame()
    f.U_table()
    U, R, res = f.get_res()
    Uy = U[-2][0]
    Ux = U[-3][0]
    erreur = abs(uy - Uy)
    print("erreur max :", erreur, "(uy = ",uy," et Uy = ",Uy," )")
    print("(ux = ", ux, " et Ux = ", Ux, " )")
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x, w, color='blue', label="E-B")
    ax.plot(X,Uy, 'o-', color='red', label="FEM")
    ax.legend()
    
    x = np.arange(len(X)) # the label locations
    labels = str(x)
    width = 0.35 # the width of the bars
    
    ax = fig.add_subplot(1, 2, 2)
    rects1 = ax.bar(x - width/2, erreur, width, label='Erreur')
    #ax.bar_label(rects1, padding=3)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Erreur')
    ax.set_title('Erreur')
    #ax.set_xticks(x, labels)
    ax.legend()
    """
    return erreur


def validation_cas4(nb_elem=1):
    """ Cas 4 : portique avec force extrémité
    
    :return: DESCRIPTION
    :rtype: TYPE
    
    """
    ### Analytique
    F = -0.1
    L = 1
    E = 210E9
    I = 0.1 * 0.1 ** 3 / 12
    uz = -2.4E-3
    
    ### FEM
    mesh = Mesh(3, [], [], debug=False)
    mesh.add_node([0, 0, 0])
    mesh.add_node([0, 0, L])
    mesh.add_node([L, 0, L])
    mesh.add_node([L, L, L])
    mesh.add_element([1, 2], "barre", "b", 10, 10)
    mesh.add_element([2, 3], "barre", "b", 10, 10)
    mesh.add_element([3, 4], "barre", "b", 10, 10)
    mesh.plot_mesh()
    f = FEM_Model(mesh, E=E)
    f.apply_load([0, 0, F, 0, 0, 0], 4)
    f.apply_bc([1, 1, 1, 1, 1, 1], 1)
    f.solver_frame()
    f.U_table()
    f.plot_forces3D(type='nodal')
    f.plot_disp_f_3D(dir='sum')
    U, R, res = f.get_res()
    Uz = U[-4][0]
    erreur = abs(uz - Uz)
    print("erreur max :", erreur, "(uz = ", uz, " et Uz = ", Uz, " )")
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x, w, color='blue', label="E-B")
    ax.plot(X,Uy, 'o-', color='red', label="FEM")
    ax.legend()
    
    x = np.arange(len(X)) # the label locations
    labels = str(x)
    width = 0.35 # the width of the bars
    
    ax = fig.add_subplot(1, 2, 2)
    rects1 = ax.bar(x - width/2, erreur, width, label='Erreur')
    #ax.bar_label(rects1, padding=3)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Erreur')
    ax.set_title('Erreur')
    #ax.set_xticks(x, labels)
    ax.legend()
    """
    return erreur

if __name__ == "__main__" :
    validation_cas4()
    
    
    """
    ERR = []
    for i in range(1,10): 
        ERR.append(validation_cas2(i))
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(1,10), ERR, color='k', label="Err")
    plt.xlabel("nombre d'elements")
    plt.ylabel("erreur")
    """
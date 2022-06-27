# -*- coding: utf-8 -*-
"""
Created on Thu May 12 18:27:25 2022

@author: ngameiro
"""

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from mesh import Mesh

def fig_maker(mesh):
    fig = mesh.plot_mesh()
    return fig

def update_fig():
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack()


# Define the window layout
# maconnerie
frame_1 = [sg.Frame(layout=[
    [sg.Text("X (m) : "), sg.Input(s=6, enable_events=True, default_text="0", key="x"),
     sg.Text("Y (m) : "), sg.Input(s=6, enable_events=True, default_text="0", key="y"),
     sg.Text("Z (m) : "), sg.Input(s=6, enable_events=True, default_text="0", key="z")],
    [sg.Button("Ajouter un noeud"), sg.Button("Supprimer un noeud")],
], title='Ajouter un noeud', title_color='red', relief=sg.RELIEF_SUNKEN)]

frame_2 = [sg.Frame(layout=[
    [sg.Text("node i : "), sg.Input(s=6, enable_events=True, default_text="1", key="ni"),
     sg.Text("node j : "), sg.Input(s=6, enable_events=True, default_text="2", key="nj")],
    [sg.Text("nom : "), sg.Input(s=6, enable_events=True, default_text="poutre", key="name"),
     sg.Text("couleur : "), sg.Input(s=6, enable_events=True, default_text="r", key="color"),
     sg.Text("hauteur (m) : "), sg.Input(s=6, enable_events=True, default_text="22", key="h"),
     sg.Text("larguer (m) : "), sg.Input(s=6, enable_events=True, default_text="10", key="b")],
    [sg.Button("Ajouter un element"), sg.Button("Supprimer un element")],
], title='Ajouter un element', title_color='red', relief=sg.RELIEF_SUNKEN)]

layout_l = [frame_1,
            frame_2,
            [sg.Button("Plot"),
             sg.Button("sauvegarder image")],
            [sg.Checkbox('Ferme symétrique', default=True, key="sym")],
            ]

layout_r = [
    [sg.Canvas(key="-CANVAS-")],
]

layout = [[sg.MenubarCustom([['File', ['Exit']], ['Edit', ['Edit Me', ]]], k='-CUST MENUBAR-', p=0)],
          [sg.T('Générateur de ferme', font='_ 16', justification='c', expand_x=True)],
          [sg.Col(layout_l), sg.Col(layout_r)],
          ]

# [sg.Radio('My first Radio!     ', "RADIO1", default=True, size=(10,1)), sg.Radio('My second Radio!', "RADIO1")]

# Create the form and show it without the plot
window = sg.Window(
    "Matplotlib Single Graph",
    layout,
    location=(0, 0),
    finalize=True,
    element_justification="center",
)
fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
matplotlib.use("TkAgg")
figure_canvas_agg = FigureCanvasTkAgg(fig, window["-CANVAS-"].TKCanvas)
figure_canvas_agg.draw()
figure_canvas_agg.get_tk_widget().pack()
sg.theme('Dark')

fig_agg = None
opened1, opened2 = True, True
mesh = Mesh(2, [], [], ax = ax, debug=True)

while True:
    # Add the plot to the window
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "Ajouter un noeud":
        x, y, z = float(values['x']), float(values['y']), float(values['z'])
        mesh.add_node([x, y])
    if event == "Ajouter un element":
        ni, nj = int(values['ni']), int(values['nj'])
        name, color, h, b = values['name'], values['color'], float(values['h']), float(values['b'])
        mesh.add_element([ni, nj], name, color, h, b)
    if event == "Plot":
        fig = fig_maker(mesh)
        update_fig()
        window.Refresh()
    if event == "sauvegarder image":
        fig.savefig('mesh.png', format='png', dpi=200)

    if event.startswith('-OPEN SEC1-'):
        opened1 = not opened1
        window['-OPEN SEC1-'].update(SYMBOL_DOWN if opened1 else SYMBOL_UP)
        window['-SEC1-'].update(visible=opened1)

    if event.startswith('-OPEN SEC2-'):
        opened2 = not opened2
        window['-OPEN SEC2-'].update(SYMBOL_DOWN if opened2 else SYMBOL_UP)
        window['-OPEN SEC2-CHECKBOX'].update(not opened2)
        window['-SEC2-'].update(visible=opened2)

window.close()

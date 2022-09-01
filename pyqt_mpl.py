import csv
import os
import sys
from os.path import dirname, realpath,join
from src.mesh import Mesh

### PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QAction, QFileDialog, QTableWidgetItem, QHeaderView
from PyQt5.QtGui import QIcon
from PyQt5.uic import  loadUiType

### Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

scriptDir=dirname(realpath(__file__))
From_Main,_= loadUiType(join(dirname(__file__),"gui.ui"))

class Sheet(QMainWindow,From_Main):
    def __init__(self):
        super(Sheet, self).__init__()
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.ToolBar()
        self.sc = myCanvas()
        self.l=QVBoxLayout(self.frame)
        self.l.addWidget(self.sc)
        self.pb_add.clicked.connect(self.add_node)
        self.pb_plot.clicked.connect(self.plot)
        self.pb_add_elem.clicked.connect(self.add_element)

        ### Maillage
        self.mesh = Mesh(dim=3)
        self.mesh.add_node([0, 0, 0])
        self.mesh.add_node([1, 2, 0])

        self.addNewContent(self.mesh.node_list, self.tw)

        self.mesh.add_element([1,2])

        self.addNewContent(self.mesh.element_list, self.elem_table)

    def ToolBar(self):
        AddFile = QAction(QIcon('images/add.png'),'Add File',self)
        AddFile.triggered.connect(self.open_sheet)
        self.toolBar= self.addToolBar('Add data File')
        self.toolBar.addAction(AddFile)
        AddPlot = QAction(QIcon('images/beam.png'),'Scatter',self)
        AddPlot.triggered.connect(self.plot)
        self.toolBar.addAction(AddPlot)

    def open_sheet(self):
        path = QFileDialog.getOpenFileName(self, "Open", "", "TXT Files (*.txt);;All Files (*)")
        if path[0]!='':
            self.FileN=path[0]

    def add_node(self):
        self.mesh.add_node([float(self.x.text()), float(self.y.text()), float(self.z.text())])
        print(" x = ",self.x.text(),"| y = ", self.y.text(),"| z = ", self.z.text())
        self.addNewContent(self.mesh.node_list, self.tw)

    def add_element(self):
        n_i = int(self.n_i.text())
        n_j = int(self.n_j.text())
        nom = self.nom.text()
        couleur = self.couleur.text()
        b = float(self.b.text())
        l = float(self.h.text())
        print(n_i, n_j, nom, couleur, b, l)
        self.mesh.add_element([n_i, n_j], nom, couleur, b, l)
        self.addNewContent(self.mesh.element_list, self.elem_table)
        #self.mesh.color
        #self.mesh.Section


    def addNewContent(self, results, table):
        header = table.horizontalHeader()
        table.clearContents()

        header.setSectionResizeMode(QHeaderView.Stretch) #ResizeToContents)
        numrows = len(results)
        numcols = len(results[0])

        table.setRowCount(numrows)
        table.setColumnCount(numcols)

        for row in range(numrows):
            for column in range(numcols):
                table.setItem(row, column, QTableWidgetItem((str(results[row][column]))))

    def plot(self):
        self.sc.plot_mesh(self.mesh)

class myCanvas(FigureCanvas):
    def __init__(self):
        self.fig=Figure()
        FigureCanvas.__init__(self,self.fig)

    def plot(self,xarray,yarray):
        self.fig.clear()
        self.ax= self.fig.add_subplot(111)
        self.ax.plot(xarray[1:],yarray[1:])
        self.ax.set_xlabel(xarray[0])
        self.ax.set_ylabel(yarray[0])
        self.draw()

    def plot_mesh(self, mesh):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        NL = mesh.node_list
        x = [x for x in NL[:, 0]]
        y = [y for y in NL[:, 1]]
        size = 10
        offset = size / 40000.
        self.ax.scatter(x, y, c='k', marker="s", s=size, zorder=5)
        color_list = []
        for i, location in enumerate(zip(x, y)):
            self.ax.annotate(i + 1, (location[0] - offset, location[1] - offset), zorder=10)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.draw()


app = QApplication(sys.argv)
sheet= Sheet()
sheet.show()
sys.exit(app.exec_())

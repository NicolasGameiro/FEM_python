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

        ### Maillage
        m1 = Mesh(dim=3)
        m1.add_node([0, 0, 0])
        m1.add_node([1, 2, 0])

        self.addNewContent(m1.node_list)

        m1.add_node([34, 2, 0])

        self.addNewContent(m1.node_list)

    def ToolBar(self):
        AddFile = QAction(QIcon('images/add.png'),'Add File',self)
        AddFile.triggered.connect(self.open_sheet)
        self.toolBar= self.addToolBar('Add data File')
        self.toolBar.addAction(AddFile)
        AddPlot = QAction(QIcon('images/beam.png'),'Scatter',self)
        AddPlot.triggered.connect(self.Plot)
        self.toolBar.addAction(AddPlot)

    def open_sheet(self):
        #QFileDialog.getOpenFileName((self,'Open CSV',),os.getenv('Home', 'CSV(*.)'))
        path = QFileDialog.getOpenFileName(self, "Open", "", "CSV Files (*.csv);;All Files (*)")
        if path[0]!='':
            self.FileN=path[0]

    def add_node(self):
        row = self.tw.rowCount()
        print(row)
        self.tw.insertRow(row)
        print(" x = ",self.x.text(),"| y = ", self.y.text(),"| z = ", self.z.text())
        self.tw.setItem(row , 0, QTableWidgetItem(self.x.text()))
        self.tw.setItem(row , 1, QTableWidgetItem(self.y.text()))
        self.tw.setItem(row , 2, QTableWidgetItem(self.z.text()))

    def add_row(self, x ,y ,z):
        row = self.tw.rowCount()
        self.tw.insertRow(row)
        self.tw.setItem(row, 0, QTableWidgetItem(x))
        self.tw.setItem(row, 1, QTableWidgetItem(y))
        self.tw.setItem(row, 2, QTableWidgetItem(z))

    def removeRow(self):
        if self.tw.rowCount() > 0:
            self.tw.removeRow(self.tw.rowCount()-1)

    def write_mesh(self, mesh):
        # clear the table widget
        self.tw.clear()
        NL = mesh.node_list
        for node in NL:
            print(*[str(n) for n in node])
            self.add_row(*[str(n) for n in node])

    def addNewContent(self,results):
        header = self.tw.horizontalHeader()
        self.tw.clearContents()

        header.setSectionResizeMode(QHeaderView.Stretch) #ResizeToContents)
        numrows = len(results)
        numcols = len(results[0])

        self.tw.setRowCount(numrows)
        self.tw.setColumnCount(numcols)

        for row in range(numrows):
            for column in range(numcols):
                self.tw.setItem(row, column, QTableWidgetItem((str(results[row][column]))))

    def Plot(self):
        f=self.FileN
        index = int(self.lineEdit.text())
        x= []
        y=[]
        with open(f, newline = '') as csv_file:
            my_file = csv.reader(csv_file, delimiter = ',',
                                 quotechar = '|')
            for row in my_file:
                x.append(str(row[0]))
                y.append(str(row[index]))
        self.sc.plot(x, y)

    def plot(self):
        x = ["x label", 2, 3]
        y = ["y lable", 3, 3]
        self.sc.plot(x, y)


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

app = QApplication(sys.argv)
sheet= Sheet()
sheet.show()
sys.exit(app.exec_())

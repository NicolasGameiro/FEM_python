import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, Button
from matplotlib.path import Path

from mesh import Mesh
from model import FEM_Model

class Index:
    ind = 0

    def __init__(self, selector):
        self.selector = selector

    def delete(self, event):
        self.ind += 1
        if len(self.selector.ind) == 0:
            print("select a node")
        else :
            print("node removed")
            print(self.selector.ind)
            print("coor :")
            print(self.selector.xys[self.selector.ind])
            mesh.del_node(self.selector.xys[self.selector.ind][-1])
            mesh.del_element(mesh.element_list[-1])
        #plt.clf()
        ax, pts = mesh.plot_mesh()
        update_plot()


    def prev(self, event):
        self.ind -= 1
        print("node modified")


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `PolygonSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
    Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
    Collection you want to select from.
    alpha_other : 0 <= float <= 1
    To highlight a selection, this tool sets all selected points to an
    alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(ax, self.onselect)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        print('\nSelected points:')
        print(selector.xys[selector.ind])
        self.canvas.draw_idle()

    def disconnect(self):
        self.poly.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

def update_plot():
    plt.subplots_adjust(bottom=0.2)
    selector = SelectFromCollection(ax, pts)

    # Génération des buttons
    callback = Index(selector)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Delete')
    bnext.on_clicked(callback.delete)
    bprev = Button(axprev, 'Modify')
    bprev.on_clicked(callback.prev)

if __name__ == '__main__':

    mesh = Mesh(2, [], [], debug=False)
    mesh.add_node([0,0])
    for i in range(1,10):
        mesh.add_node([i, 0])
        mesh.add_element([i, i+1], "entrait", "r", 4, 2, 5)
    ax, pts = mesh.plot_mesh()

    plt.subplots_adjust(bottom=0.2)
    """
    grid_size = 5
    grid_x = np.arange(grid_size)
    grid_y = np.ones(grid_size)
    pts = ax.scatter(grid_x, grid_y)
    ax.plot(grid_x, grid_y, '--')
    """

    selector = SelectFromCollection(ax, pts)

    # Génération des buttons
    callback = Index(selector)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Delete')
    bnext.on_clicked(callback.delete)
    bprev = Button(axprev, 'Modify')
    bprev.on_clicked(callback.prev)

    print("Select points in the figure by enclosing them within a polygon.")
    print("Press the 'esc' key to start a new polygon.")
    print("Try holding the 'shift' key to move all of the vertices.")
    print("Try holding the 'ctrl' key to move a single vertex.")

    plt.show()

    selector.disconnect()

    # After figure is closed print the coordinates of the selected points
    print('\nSelected points:')
    print(selector.xys[selector.ind])
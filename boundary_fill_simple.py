# NREL 2020
# Patrick Duffy
# Script to fill boundaries with turbines using a basic filling algorithm
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def simple_boundary_fill(boundaries, D, spacing):
    """ Simplest approach to filling an area defined by polygon vertices 
    with turbines at a uniform grid spacing. Returns turbine x, y coordinates"""
    # Initialize the layout lists
    layout_x = []
    layout_y = []

    # Find the x_min, x_max, y_min, y_max
    b_x = []
    b_y = []
    for i in boundaries:
        b_x.append(i[0])
        b_y.append(i[1])
    x_min = min(b_x)
    x_max = max(b_x)
    y_min = min(b_y)
    y_max = max(b_y)

    print(x_max, y_max)
    # Make query points
    s = D * spacing
    n_x = math.floor((x_max - x_min)/(D * spacing))
    n_y = math.floor((y_max - y_min)/ s)

    # Check each query position starting at xmin, ymin
    for ii in range(n_x):
        x_i = x_min + ii * s

        for jj in range(n_y):
            y_i = y_min + jj * s
            if _point_inside_polygon(x_i, y_i, boundaries):
                layout_x.append(x_i)
                layout_y.append(y_i)

    return layout_x, layout_y

def _point_inside_polygon(x, y, poly):
        n = len(poly)
        inside =False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

def plot_layout_results(boundaries, layout_x,  layout_y, D):
        """
        Method to plot the turbine locations and boundary.
        """
        fig = plt.figure(figsize=(9,6))
        fontsize= 16
        buffer = 5000.
        r = D/2
        r_list = len(layout_x)*[r]

        ax = fig.add_subplot(111, aspect='equal')
        ax.axis([min(layout_x)-buffer, max(layout_x)+buffer, min(layout_y)-buffer, max(layout_y)+buffer])
        for x, y, r in zip(layout_x, layout_y, r_list):
            ax.add_artist(Circle(xy=(x, y), 
                        radius=r))

        ax.set_xlabel('x (m)', fontsize=fontsize)
        ax.set_ylabel('y (m)', fontsize=fontsize)

        verts = boundaries
        for i in range(len(verts)):
            if i == len(verts)-1:
                plt.plot([verts[i][0], verts[0][0]], \
                         [verts[i][1], verts[0][1]], 'r')        
            else:
                plt.plot([verts[i][0], verts[i+1][0]], \
                         [verts[i][1], verts[i+1][1]], 'r')
        plt.show()

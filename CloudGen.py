"""
All the codes presented below were developed by:
    Dr. Gerardo Tinoco Guerrero
    Universidad Michoacana de San Nicolás de Hidalgo
    gerardo.tinoco@umich.mx

With the funding of:
    National Council of Humanities, Sciences, and Technologies, CONAHCyT (Consejo Nacional de Humanidades, Ciencias y Tecnologías, CONAHCyT). México.
    Coordination of Scientific Research, CIC-UMSNH (Coordinación de la Investigación Científica de la Universidad Michoacana de San Nicolás de Hidalgo, CIC-UMSNH). México
    Aula CIMNE-Morelia. México

Date:
    April, 2023.

Last Modification:
    October, 2023.
"""

import dmsh
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint
plt.rcParams.update({'font.size': 26})

def CreateCloud(xb, yb, num):
    '''
    This function generates a cloud of points within a specified boundary defined by the (xb, yb) coordinates.
    It calculates the maximum distance between boundary nodes and uses it to determine node spacing.
    The resulting point cloud contains nodes within the boundary, with boundary nodes marked as 1 in the third column of X.

    Parameters:
        xb          ndarray         Array of x-coordinates of boundary nodes.
        yb          ndarray         Array of y-coordinates of boundary nodes.
        num         int             Number of desired nodes in the cloud.

    Returns:
        X           ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        cells       ndarray         Array representing the mesh cells generated.
    
    Example:
        xb       = np.array([0.0, 1.0, 1.0, 0.0])
        yb       = np.array([0.0, 0.0, 1.0, 1.0])
        num      = 100
        X, cells = CreateCloud(xb, yb, num)
    '''

    # Find the maximum distance between the boundary nodes.
    dist  = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2))                    # Calculate the distance between nodes based on the maximum distance.
    dist /= num                                                                     # Calculate the distance between nodes based on the maximum distance.

    # Create the Triangulation
    pb       = np.column_stack((xb, yb))                                            # Stack xb and yb arrays as columns to create point coordinates.
    geo      = dmsh.Polygon(pb)                                                     # Create a Polygon object from the boundary points.
    X, cells = dmsh.generate(geo, dist)                                             # Generate a triangulation using dmsh.

    # Create a polygon
    poly = Polygon(pb).buffer(-dist/4)                                              # Create a buffer zone around the boundary.

    # Check if points are within the buffer zones
    points      = [Point(point[0], point[1]) for point in X]                        # Convert the cloud points to Shapely Point objects.
    inside_poly = np.array([i.within(poly) for i in points])                        # Check wether points are inside the buffer zone.

    # Add a third column to X and mark boundary nodes as 1
    X = np.column_stack((X, 1 - inside_poly.astype(int)))                           # Add a column to X to mark boundary nodes.

    return X, cells

def CreateCloud_Holes(xb, yb, num, h_coor):
    '''
    This function generates a cloud of points within a specified boundary defined by the (xb, yb) coordinates.
    It calculates the maximum distance between boundary nodes and uses it to determine node spacing.
    The resulting point cloud contains nodes within the boundary, with boundary nodes marked as 1 in the third column of X.

    Parameters:
        xb          ndarray         Array of x-coordinates of boundary nodes.
        yb          ndarray         Array of y-coordinates of boundary nodes.
        num         int             Number of desired nodes in the cloud.
        h_coor      ndarray         Array with the coordinates and radius for the holes.
                                    (x-coordinate, y-coordinate, radius)

    Returns:
        X           ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        cells       ndarray         Array representing the mesh cells generated.
    
    Example:
        xb       = np.array([0.0, 1.0, 1.0, 0.0])
        yb       = np.array([0.0, 0.0, 1.0, 1.0])
        num      = 100
        h_coor   = [(0.5, 0.3, 0.05)]
        X, cells = CreateCloud_Holes(xb, yb, num, h_coor)
    '''

    # Find the maximum distance between the boundary nodes.
    dist  = np.max(np.sqrt(np.diff(xb.T)**2 + np.diff(yb.T)**2))                    # Calculate the distance between nodes based on the maximum distance.
    dist /= num                                                                     # Calculate the distance between nodes based on the maximum distance.

    # Create the Triangulation
    pb       = np.column_stack((xb, yb))                                            # Stack xb and yb arrays as columns to create point coordinates.
    geo      = dmsh.Polygon(pb)                                                     # Create a Polygon object from the boundary points.

    # Create holes by subtracting circles from the polygon
    for hole_x, hole_y, hole_radius in h_coor:                                      # Take the coordinates and radius of all the holes.
        geo -= dmsh.Circle([hole_x, hole_y], hole_radius)                           # Subtract the holes from the Polygon.

    X, cells = dmsh.generate(geo, dist)                                             # Generate a triangulation using dmsh.

    # Create a polygon and hole buffers
    poly         = Polygon(pb).buffer(-dist/4)                                      # Create a buffer zone around the boundary.
    hole_buffers = [Point(hole_x, hole_y).buffer(hole_radius).buffer(dist/4) for hole_x, hole_y, hole_radius in h_coor]
                                                                                    # Create a buffer zone around each hole.

    # Initialize an array A to store markings for each point
    points = [Point(point[0], point[1]) for point in X]                             # Convert the cloud points to Shapely Point objects.
    A      = np.zeros([len(points),1])                                              # Initialize A to store the markings.

    # Mark points as 1 if they are outside the polygon, 2 if they are inside any of the hole buffers
    for i, point in enumerate(points):                                              # For each of the points.
        if not point.within(poly):                                                  # Check wether points are not inside the buffer zone of the external boundary.
            A[i] = 1                                                                # Add 1 to that point.
        else:                                                                       # If it is not in the external boundary.
            for j, hole_buffer in enumerate(hole_buffers):                          # For each of the holes.
                if point.within(hole_buffer):                                       # Check wether points are not inside the buffer zone of the holes.
                    A[i] = 2                                                        # Add 2 to that point.
                    break

    # Add a third column to X and mark boundary nodes as 1
    X = np.column_stack((X, A))                                                     # Add a column to X to mark boundary nodes.

    return X, cells

def GraphCloud(p, save = False, folder = '', nom = '', show = True):
    """
    This function creates a scatter plot of a cloud of points where points are colored based on the position in the cloud:
    Blue markers are used for inner points, and red markers for boundary points.
    The plot is saved as PNG and EPS files with the specified name and displayed to the screen.

    Parameters:
        p           ndarray         Array with the coordinates of the cloud of points with boundary node markings.
        nom         str             Name for the generated plot and file.
        show        bool            Flag to indicate whether the figure should be shown on screen.
        holes       bool            Flag to indicate whether holes are present.

    Returns:
        None
    
    Example:
        p = np.array([[1.0, 2.0, 0],
                      [2.0, 3.0, 1],
                      [3.0, 4.0, 0]])
        GraphCloud(p, "example_plot")
    """

    # Parameter initialization
    if save:
        nomp = folder + nom + '.png'                                                # Define the file name for the saved plot.
        nome = folder + nom + '.eps'                                                # Define the file name for the saved plot.
    else:
        show = True

    color = ['blue' if x == 0 else 'red' for x in p[:,2]]                           # Determine marker color based on the third column.

    # Cloud plot
    plt.rcParams["figure.figsize"] = (16, 12)                                       # Configure plot size.
    plt.scatter(p[:,0], p[:,1], c=color, s = 20)                                    # Create scatter plot.
    plt.title(nom)                                                                  # Set plot title.
    plt.axis('equal')
    if show:
        plt.show()                                                                  # Display the plot.
    if save:
        plt.savefig(nomp)                                                           # Save the plot as a PNG file.
        plt.savefig(nome, format='eps', bbox_inches='tight')                        # Save the plot as a EPS file.
    plt.close()                                                                     # Close the plot to release resources.

def Randomize(p, me):

    ## Variable Initialization.
    m        = len(p[:, 0])                                                         # Get the number of boundary nodes.
    boundary = MultiPoint(p[p[:, 2] == 1][:, :2])                                   # Get the boundary nodes.
    pol      = boundary.convex_hull                                                 # Create a polygon.

    ## Check how big the movement should be.
    if me == '1':
        r = 0.05
    elif me == '2':
        r = 0.02
    elif me == '3':
        r = 0.01
    
    ## Randomly move the nodes.
    for i in range(0, m):
        if p[i, 2] == 0:
            inside_poly = False
            while not inside_poly:
                move_x   = p[i, 0]*random.uniform(-r, r)
                move_y   = p[i, 1]*random.uniform(-r, r)

                new_point = Point(p[i, 0] + move_x, p[i, 1] + move_y)
                inside_poly = pol.contains(new_point)

                if inside_poly:
                    p[i, 0] += move_x
                    p[i, 1] += move_y

    return p
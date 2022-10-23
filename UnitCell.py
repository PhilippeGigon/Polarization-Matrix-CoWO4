from asyncio import constants
import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import Constants
from P_matrix import rotate_to_experimental
from mpl_toolkits import mplot3d
from AtomGenerator import Atom
#####################################################
# linewidth
lw = 3
# markersize
ms = 100
#####################################################
# Unit cell parameters in Angstroem
a_cell = Constants.a
b_cell = Constants.b
c_cell = Constants.c
# Angle in rad
beta_cell = 2*math.pi/360*Constants.beta
#####################################################


def plotc(self):
    '''Plots the unit cell without atoms'''
    plt.plot([0, self.a[0]], [0, self.a[1]], [
        0, self.a[2]], linewidth=lw, c="Black")
    plt.plot([0, self.b[0]], [0, self.b[1]], [
        0, self.b[2]], linewidth=lw, c="Black")
    plt.plot([0, self.c[0]], [0, self.c[1]], [
        0, self.c[2]], linewidth=lw, c="Black")
    plt.plot([self.a[0], self.a[0]+self.b[0]], [self.a[1], self.b[1]+self.a[1]], [
        self.a[2], self.a[2]+self.b[2]], linewidth=lw, c="Black")
    plt.plot([self.b[0], self.b[0]+self.c[0]], [self.b[1], self.b[1] +
                                                self.c[1]], [self.b[2], self.b[2]+self.c[2]], linewidth=lw, c="Black")
    plt.plot([self.b[0]+self.c[0], self.b[0]+self.c[0]+self.a[0]], [self.b[1] +
                                                                    self.c[1], self.a[1]+self.b[1] +
                                                                    self.c[1]], [self.b[2]+self.c[2], self.b[2]+self.c[2]+self.a[2]], linewidth=lw, c="Black")
    plt.plot([self.a[0]+self.c[0], self.b[0]+self.c[0]+self.a[0]], [self.a[1] +
                                                                    self.c[1], self.a[1]+self.b[1] +
                                                                    self.c[1]], [self.a[2]+self.c[2], self.b[2]+self.c[2]+self.a[2]], linewidth=lw, c="Black")
    plt.plot([self.a[0], self.a[0]+self.c[0]], [self.a[1], self.c[1] +
                                                self.a[1]], [self.a[2], self.c[2]+self.a[2]], linewidth=lw, c="Black")
    plt.plot([self.a[0]+self.b[0], self.a[0]+self.c[0]+self.b[0]], [self.a[1]+self.b[1], self.c[1] + self.b[1]
                                                                    + self.a[1]], [self.a[2]+self.b[2], self.c[2]+self.b[2]+self.a[2]], linewidth=lw, c="Black")
    plt.plot([self.c[0], self.c[0]+self.b[0]], [self.c[1], self.c[1] +
                                                self.b[1]], [self.c[2], self.c[2]+self.b[2]], linewidth=lw, c="Black")
    plt.plot([self.b[0], self.a[0]+self.b[0]], [self.b[1], self.a[1] +
                                                self.b[1]], [self.b[2], self.a[2]+self.b[2]], linewidth=lw, c="Black")
    plt.plot([self.c[0], self.a[0]+self.c[0]], [self.c[1], self.a[1] +
                                                self.c[1]], [self.c[2], self.a[2]+self.c[2]], linewidth=lw, c="Black")

#####################################################


def plotmagstruct(self):
    '''This method plots the magnetic structure in the repeated
    scheme, i.e. multiple cells in c direction such that the whole
    periodicity is represented'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    self.plot_unit_cell()
    # Turns of the color of the grid
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_zlim([-c_cell, c_cell])
    ax.set_ylim([-b_cell, b_cell])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plots the atomic position with the mag moment
    for l in self.atomic_positions_repeated:
        ax.scatter(l[0][0], l[0][1], l[0][2],
                   c=Constants.colors[l[1]], s=ms)
        # plots the real part of the magnetic moments starting at the atomic positions
        plt.plot([l[0][0], l[0][0]+(l[2][0]).real], [l[0][1], l[0][1]+(l[2][1]).real],
                 [l[0][2], l[0][2]+(l[2][2]).real], linewidth=lw, c=Constants.colors[l[1]])
    plt.show()
#####################################################


def magmom(self, Q):
    '''This method gives the effect of all the magnetic 
    moments on neutron with Q=kin-kout the scattering vector'''
    m = np.array([0, 0, 0])
    Q = Q[0]*self.arecip+Q[1]*self.brecip+Q[2]*self.crecip

    for k in self.atomic_positions_repeated:
        r = k[0]
        rdotQ = np.dot(r, Q)
        m = m+k[2]*cmath.exp(1j*rdotQ)
    return m
#####################################################


class Unit_cell:
    def __init__(self, atoms: Atom, multiplicity_):
        '''An array of atoms to initialize. The param
        multiplicity is the number of cells in z direction
        allowing to plot all of the magnetic unit cell'''

        # Number of cells in c direction
        self.multiplicity = multiplicity_

        # Real lattice vectors, monoclinic
        self.a = a_cell*np.array([1, 0, 0])
        self.b = b_cell*np.array([0, 1, 0])
        self.c = c_cell * \
            np.array([-math.sin(beta_cell-math.pi/2),
                     0, math.cos(beta_cell-math.pi/2)])

        # reciprocal lattice vectors:
        self.arecip = 2*math.pi * \
            np.cross(self.b, self.c)/(np.dot(self.a, np.cross(self.b, self.c)))
        self.brecip = 2*math.pi * \
            np.cross(self.c, self.a)/(np.dot(self.b, np.cross(self.c, self.a)))
        self.crecip = 2*math.pi * \
            np.cross(self.a, self.b)/(np.dot(self.c, np.cross(self.a, self.b)))

        # This vector contains Tuples of the following form:
        # (np.array([x,y,z]),index), with [x,y,z] the position,
        # index the identifier of the atom. Representing the
        # primitive cell
        self.atomic_positions = list()
        for i in atoms:
            # Computes the position in x,y,z basis
            pos = i.position[0]*self.a+i.position[1] * \
                self.b+i.position[2]*self.c

            # Checks that we dont add twice the same atom
            isinlist = False
            for k, j in self.atomic_positions:
                if all(k == pos):
                    isinlist = True
                    break
            if not isinlist:
                self.atomic_positions.append(
                    (pos, i.element_index))
            else:
                raise ValueError("Two atoms at the same position")

    ##################################################################
        # This vector contains Tuples of the following form:
        # (np.array([x,y,z]),index,np.array([m1,m2,m3])), with [x,y,z] the position,
        # index the identifier of the atom and [m1,m2,m3] the magnetic moment
        # primitive cell + cells in z direction
        self.atomic_positions_repeated = list()
        for i in atoms:
            for k in range(0, self.multiplicity):
                # Computes position in abc basis
                posabc = np.array(
                    [i.position[0]+k, i.position[1], i.position[2]])
                # Computes the position in x,y,z basis
                pos = posabc[0]*self.a+posabc[1] * \
                    self.b+posabc[2]*self.c

                # Computes the magnetic moment in realspace basis
                magmom = i.magnetic_struct.get_m(posabc)

                # magnetic moment in x, y, z basis
                magmom_xyz = magmom[0]*self.a+magmom[1]*self.b+magmom[2]*self.c

                # We don't have to worry to add twice, as already tested above
                self.atomic_positions_repeated.append(
                    (pos, i.element_index, magmom_xyz))

    ##################################################################
    get_M = magmom
    plot_unit_cell = plotc
    show_magnetic_structure = plotmagstruct

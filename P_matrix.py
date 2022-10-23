import cmath
import math
from os import major
import matplotlib
import pylab as plt
import numpy as np
import MagneticStructure
import UnitCell
import Constants
import fractions

#####################################################
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "17"
#####################################################
# angle in rad between a and c
beta_cell = 2*math.pi/360*Constants.beta
#####################################################


def plot_all_P(P, ya, Q, B):
    fig, ax = plt.subplots()
    # bar width
    w = 0.5
    ax.set_ylabel('$P_{ij}$')
    height = []
    y = []

    for k in range(len(P)):
        height += [P[k][0][0], P[k][0][1], P[k][0][2], P[k][1][0],
                   P[k][1][1], P[k][1][2], P[k][2][0], P[k][2][1], P[k][2][2], 0]
        y += [ya[k][0], ya[k][1], ya[k][2], ya[k][3], ya[k][4],
              ya[k][5], ya[k][6], ya[k][7], ya[k][8], cmath.nan]
    # x-coordinates of points
    x = np.linspace(0, 25, 10*5)
    major_tick = [x[9], x[19], x[29], x[39], x[49]]
    minor_tick = [(x[4]), x[14], x[24], x[34], x[44]]
    ax.set_ylim([-1, 1])
    ax.set_xlim([-w, 9*w*5+w/2])
    ax.set_xticks(major_tick)
    ax.set_xticks(minor_tick, minor=True)
    ax.set_xticks(major_tick)
    ax.set_xticks(minor_tick, minor=True)
    ax.set_yticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], minor=False)
    ax.set_yticklabels(["-1", "", "-0.5", "", "0", "",
                       "0.5", "", "1"])

    ax.grid(which='major', alpha=0.3)
    xlabels = [r'$\frac{1}{2}00$', r'$\frac{3}{2}00$',
               r'$\frac{1}{2}\bar{1}0$', r'$\frac{1}{2}\bar{2}0$', r'$\frac{3}{2}\bar{1}0$']
    ax.set_xticklabels(xlabels, minor=True)
    ax.set_xticklabels([], minor=False)
    model = ax.bar(x,
                   height,
                   width=w,
                   color=(2/3, 2/3, 2/3, 2/3), linewidth=1.2, label='Model')
    e = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0]
    ee = []
    ee += e
    ee += e
    ee += e
    ee += e
    ee += e
    data = ax.errorbar(x, y, yerr=ee, color="black", fmt="o",
                       markersize=3.5, linewidth=0.5, label='Data')
    plt.legend(handles=[model, data], fancybox=True, framealpha=1)

    xlines = [x[9], x[19], x[29], x[39]]
    plt.vlines(xlines, -1, 1, colors='k', linestyles='solid', linewidth=0.5)
    plt.hlines(0, -100, 100, colors='k',
               linestyles='solid', linewidth=0.5)

    if B:
        plt.show()
    if not B:
        name = "CoWO4_Polarization"
        plt.tight_layout()
        plt.savefig(name+".png", format='png',
                    dpi=1000, bbox_inches='tight')
        plt.close(fig)

#####################################################


def plot_P(P, y, Q, B):
    '''This method plots the P matrix as bar chart
    with the data points given by the vector y. If B=True
    the plot is shown, otherwise saved as image.'''
    Q = np.array2string(Q)
    height = [P[0][0], P[0][1], P[0][2], P[1][0],
              P[1][1], P[1][2], P[2][0], P[2][1], P[2][2]]
    bars = ('XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ')
    x_pos = np.arange(len(bars))
    # bar width
    w = 0.5
    # x-coordinates of points
    x = np.linspace(1, 1+9*w, 9)
    fig, ax = plt.subplots()
    fig.suptitle("Q = " + Q, fontsize=16)
    plt.ylim([-1, 1])
    ax.bar(x,
           height,
           width=w,
           tick_label=bars,
           color=(2/3, 2/3, 2/3, 2/3), edgecolor=(0, 0, 0, 2/3))
    for i in range(len(x)):
        ax.scatter(x[i], y[i], color="black")

    name = "CoWO4 "+str(Q)
    if B:
        plt.show()
    if not B:
        plt.tight_layout()
        plt.savefig(name+".pdf", format='pdf', dpi=1000)
        plt.close(fig)

#####################################################


def rotate_to_experimental(v, Qabc):
    '''Takes a vector in the x',y',z' basis of the crystal
    (i.e. a=x',b=y', c linear comb of x',z') and gives the
    vector in the experimental basis i.e. Q=x'''
    # Rotation around y axis to align c with z
    a_cryst = Constants.a*np.array([1, 0, 0])
    b_cryst = Constants.b*np.array([0, 1, 0])
    c_cryst = Constants.c*np.array([0, 0, 1])

    Q = a_cryst*Qabc[0]+b_cryst*Qabc[1]+c_cryst*Qabc[2]
    Q = Q/math.sqrt(np.dot(Q, Q))
    # thetax = math.pi
    # Rx = np.array([[1, 0, 0], [0, math.cos(thetax), -math.sin(thetax)],
    #               [0, math.sin(thetax), math.cos(thetax)]])
    # Angle of rotation (s.t. Q parallel to x)
    theta = -math.acos(np.dot(Q, np.array([1, 0, 0])))
    Rz = np.array([[math.cos(theta), -math.sin(theta), 0],
                   [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    R = Rz
    return np.dot(R, v)
#####################################################


def get_P(unitcell, Q, p0):
    '''Takes a UnitCell and computes the polarization
    matrix P for vector Q==x in experimental frame'''
    # computes total magnetic moment in xyz coord of the crystal
    mcryst = unitcell.get_M(Q)

    # Transforms to the experimental frame
    M = rotate_to_experimental(mcryst, Q)

    # Gets part of M perpendicular to Q = (1,0,0)
    Mperp = M-np.array([1, 0, 0])*M[0]

    ###################################
    Mnorm2 = (np.dot(Mperp.conjugate(), Mperp)).real
    Mnorm2y = (Mperp[1].conjugate()*Mperp[1]).real
    Mnorm2z = (Mperp[2].conjugate()*Mperp[2]).real
    Ryz = 2*(Mperp[1]*Mperp[2].conjugate()).real
    Jyz = 2*(Mperp[1]*Mperp[2].conjugate()).imag
    Ix = Mnorm2+p0*Jyz
    Iy = Mnorm2
    Iz = Mnorm2
    Pxx = (-p0*Mnorm2-Jyz)/Ix
    Pxy = (-Jyz)/Iy
    Pxz = (-Jyz)/Iz
    Pyx = 0
    Pyy = p0*(Mnorm2y-Mnorm2z)/Iy
    Pyz = p0*Ryz/Iz
    Pzx = 0
    Pzy = p0*Ryz/Iy
    Pzz = p0*(-Mnorm2y+Mnorm2z)/Iz
    P = np.array([[Pxx, Pxy, Pxz],
                  [Pyx, Pyy, Pyz], [Pzx, Pzy, Pzz]])
    ###################################

    return P

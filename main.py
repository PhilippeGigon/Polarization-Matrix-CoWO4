import numpy as np
import math
import AtomGenerator
import UnitCell
import MagneticStructure
import P_matrix
#####################################################


def main():
    ##########################################
    ##########Magnetic structures#############
    ##########################################
    """This code does the computation for the Cobalt with 
    magnetic propagation vector q = (1/2,0,0)"""
    # Spin direction in abc basis
    m1 = np.array([-1, 0, -1.05])
    Flip1 = MagneticStructure.Flip(m1)
    multiplicity = 2

    #######################################
    ##########Generating atoms#############
    #######################################
    # Cobalt sites
    Co1 = AtomGenerator.Atom(np.array([0.5, 0.6712, 0.25]), 'Co', Flip1)
    Co2 = AtomGenerator.Atom(np.array([0.5, 0.3288, 0.75]), 'Co', Flip1)
    atoms = [Co1, Co2]
    CoWO4 = UnitCell.Unit_cell(atoms, multiplicity)
    CoWO4.show_magnetic_structure()

    ##########################################
    ##########Polarization Matrix#############
    ##########################################
    # Bragg reflection with scatering vectors Qi
    # yi are the experimentaly measured values

    Q1 = np.array([1/2, 1, 0])
    p01 = 0.8518
    P1 = P_matrix.get_P(CoWO4, Q1, p01)
    y1 = [-0.8518, 0.0191, 0.0452, -0.0342, -
          0.1961, -0.8233, 0.0074, -0.8228, 0.1694]

    Q2 = np.array([3/2, 1, 0])
    p02 = 0.8809
    P2 = P_matrix.get_P(CoWO4, Q2, p02)
    y2 = [-0.8809, 0.0498, 0.0505, -0.0405, -
          0.5177, -0.6755, -0.0025, -0.6804, 0.5317]

    Q3 = np.array([1/2, 2, 0])
    p03 = 0.8725
    P3 = P_matrix.get_P(CoWO4, Q3, p03)
    y3 = [-0.8725, 0.0099, 0.0320, -0.0293, -
          0.0931, -0.8586, 0.0392, -0.8569, 0.0985]

    Q4 = np.array([3/2, 0, 0])
    p04 = 0.8839
    P4 = P_matrix.get_P(CoWO4, Q4, p04)
    y4 = [-0.8839, 0.0680, 0.0692, -0.0258, -
          0.8508, -0.2558, -0.0615, -0.2384, 0.8572]

    Q5 = np.array([1/2, 0, 0])
    p05 = 0.8675
    P5 = P_matrix.get_P(CoWO4, Q5, p05)
    y5 = [-0.8675, 0.0401, 0.0110, -0.0107,
          -0.8479, -0.1816, -0.1221, -0.1794, 0.8402]

    ###############################################
    ##########Ploting all the matrices#############
    ###############################################

    Pall = np.array([P5, P4, P1, P3, P2])
    yall = np.array([y5, y4, y1, y3, y2])
    Qall = np.array([Q5, Q4, Q1, Q3, Q2])
    P_matrix.plot_all_P(Pall, yall, Qall, True)


if __name__ == '__main__':
    main()

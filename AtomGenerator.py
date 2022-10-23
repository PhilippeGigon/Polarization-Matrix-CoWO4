import numpy as np
import Constants
import MagneticStructure
#####################################################


class Atom:
    def __init__(self, pos, name_, MS_):
        '''pos is the position of the atom, name_ its name 
        and MS_ is the magnetic structure cf MagneticStructure.py'''

        if not name_ in Constants.elements:
            raise ValueError("Unknown element")
        self.element_index = Constants.elements.index(name_)

        # nuclear Scattering length
        self.b = Constants.scattlength[self.element_index]

        # Position of the first atom, on basis of unit cell!
        # i.e. (1,2,3)=1*a+2*b+3*c
        self.position = pos
        self.magnetic_struct = MS_
#####################################################


def move2Unitcell(pos):
    '''This function takes a vector and moves it to the 
    primitive cell by translation of Bravais vectors '''
    for k in range(0, 3):
        if pos[k] < 0:
            while pos[k] < 0:
                pos[k] = pos[k]+1
        if pos[k] > 1:
            while pos[k] > 1:
                pos[k] = pos[k]-1
    return pos
#####################################################

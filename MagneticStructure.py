import numpy as np
import math
import cmath
#####################################################


class Flip:
    '''This class encodes the CoWO4 structure where the 
    spin flips after one unit cell.'''

    def __init__(self, m1):
        self.m = m1

    def get_m(self, r):
        '''This function returns the magnetic moment
        at a given position r (on basis a,b,c) for the flip with q=(1/2,0,0)'''
        if (r[0] < 1) and (r[1] < 1) and (r[2] < 1):
            return self.m
        else:
            return -self.m

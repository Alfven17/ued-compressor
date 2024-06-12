""" This module implements the first and second transfer matrix for common beamline elements. 
"""
import numpy as np
from numpy import sin, cos, tan, sinh, cosh, sqrt, zeros, power
from functools import reduce

Gamma0 = 9.241
print(f"Global variable Gamma0 set to {Gamma0}")

def Drift(s):
    R = np.array([[1, s, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, s, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, s/Gamma0**2],
                  [0, 0, 0, 0, 0, 1]
                 ])
    T = zeros((6,6,6))
    T[4,1,1] = - s / 2
    T[4,3,3] = - s / 2
    
    return (R, T)

def DriftSections(s, dz=0.1):
    """
    divide drift into sections, and return a list of drifts
    """
    return [Drift(dz) if i < int(s // dz) else Drift(s % dz) for i in range(int(np.ceil(s / dz)))]    
    
def ThinTDC(k, direction='y'):
    if direction == 'x':
        R = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, k, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [k, 0, 0, 0, 0, 1]
                     ])
    elif direction == 'y':
        R = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, k, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, k, 0, 0, 1]
                     ])
                     
    T = zeros((6,6,6))
    return (R, T)
    

def Solenoid(K, L):
    """
    K = B_{0} / (2 * p / e)
    1/f = K * S = K * sin(K*L) 
    second order element from Elegant compute_matrices.c
    The K in elegant is B_{0} / (B * \rho) so it should be twice the value here. 
    """
    C = cos(K * L)
    S = sin(K * L)
    R = np.array([[C*C, S*C/K, S*C, S*S/K, 0, 0],
                  [-K*S*C, C*C, -K*S*S, S*C, 0, 0],
                  [-S*C, -S*S/K, C*C, S*C/K, 0, 0],
                  [K*S*S, -S*C, -K*S*C, C*C, 0, 0],
                  [0, 0, 0, 0, 1, L/Gamma0**2],
                  [0, 0, 0, 0, 0, 1]
                 ])
    
    T = zeros((6,6,6))
    
    C2 = cos(2 * K * L)
    S2 = sin(2 * K * L)
    
    T[0,5,0] = K * L * S2
    T[1,5,1] = K * L * S2
    T[2,5,2] = K * L * S2
    T[3,5,3] = K * L * S2
    
    T[0,5,1] = S2 / 2 / K - L * C2
    T[2,5,3] = S2 / 2 / K - L * C2
    
    T[0,5,2] = - K * L * C2
    T[1,5,3] = - K * L * C2
    T[3,5,1] = K * L * C2
    T[2,5,0] = K * L * C2
    
    T[2,5,1] =  - ( 1 - C2 ) / 2 / K +  L * S2
    T[0,5,3] =    ( 1 - C2 ) / 2 / K - L * S2
    T[1,5,0] = 0.5 * K * (2 * K * L * C2 + S2)
    T[3,5,2] = 0.5 * K * (2 * K * L * C2 + S2)
    T[1,5,2] = 0.5 * K * (1 - C2 + 2 * K * L * S2)
    T[3,5,0] = - 0.5 * K * (1 - C2 + 2 * K * L * S2)
    
    T[4,1,1] = - L / 2
    T[4,3,3] = - L / 2
    T[4,0,0] = - K * K * L / 2
    T[4,2,2] = - K * K * L / 2
    T[4,3,0] = K * L
    T[4,2,1] = - K * L
    
    return (R, T)
    
def Rotate(alpha):
    """
    alpha in radian
    """
    C = cos(alpha)
    S = sin(alpha)
    R = np.array([[C,0,S,0,0,0],
                  [0,C,0,S,0,0],
                  [-S,0,C,0,0,0],
                  [0,-S,0,C,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]])
    T = zeros((6,6,6))
    return R, T
    
    
def Quad(K1, s):
    """
    K1 = B_{0} / a / Brho = G / Brho
    """
    k = sqrt(abs(K1))
    T = zeros((6,6,6))
    if K1 > 0:
        R = np.array([[cos(k*s), sin(k*s)/k,0,0,0,0],
                      [-k*sin(k*s), cos(k*s),0,0,0,0],
                      [0,0,cosh(k*s), sinh(k*s)/k,0,0],
                      [0,0,k*sinh(k*s), cosh(k*s),0,0],
                      [0,0,0,0,1,s/Gamma0**2],
                      [0,0,0,0,0,1]
        ])
    
        T[0,5,0] = 1 / 2 * k * s * sin(k*s)
        T[0,5,1] = (-3 * k**4 * s * cos(k*s) + 3 * k**3 * sin(k*s)) / 6 / k**4
        T[1,5,0] = ( 3 * k**4 * s * cos(k*s) + 3 * k**3 * sin(k*s)) / 6 / k**2
        T[1,5,1] = 1 / 2 * k * s * sin(k*s)
        
        T[2,5,2] = - 1 / 2 * k * s * sinh(k*s)
        T[2,5,3] = (-k**2 * s * cosh(k*s) + k * sinh(k*s)) / 2 / k**2
        T[3,5,2] = - 1 / 2 * (k**2 * s * cosh(k*s) + k * sinh(k*s))
        T[3,5,3] = - 1 / 2 * k * s * sinh(k*s)
        
        T[4,0,0] = - 1 / 4 * k**2 * (s - cos(k*s)*sin(k*s)/k)
        T[4,1,0] = 1 / 2 * sin(k*s)**2
        T[4,1,1] = - 1 / 4 * (s + cos(k*s)*sin(k*s)/k)
        
        T[4,2,2] = 1 / 4 * k**2 * (s - cosh(k*s)*sinh(k*s)/k)
        T[4,3,2] = - 1 / 2 * sinh(k*s)**2
        T[4,3,3] = - 1 / 4 * (s + cosh(k*s)*sinh(k*s)/k)
        
        
    elif K1 < 0:
        R = np.array([[cosh(k*s), sinh(k*s)/k,0,0,0,0],
                      [k*sinh(k*s), cosh(k*s),0,0,0,0],
                      [0,0,cos(k*s), sin(k*s)/k,0,0],
                      [0,0,-k*sin(k*s), cos(k*s),0,0],
                      [0,0,0,0,1,s/Gamma0**2],
                      [0,0,0,0,0,1]
        ])
        T[0,5,0] = - 1 / 2 * k * s * sinh(k*s)
        T[0,5,1] = (-k**2 * s * cosh(k*s) + k * sinh(k*s)) / 2 / k**2
        T[1,5,0] = - 1 / 2 * (k**2 * s * cosh(k*s) + k * sinh(k*s))
        T[1,5,1] = - 1 / 2 * k * s * sinh(k*s)
        
        T[2,5,2] = 1 / 2 * k * s * sin(k*s)
        T[2,5,3] = (-3 * k**4 * s * cos(k*s) + 3 * k**3 * sin(k*s)) / 6 / k**4
        T[3,5,2] = ( 3 * k**4 * s * cos(k*s) + 3 * k**3 * sin(k*s)) / 6 / k**2
        T[3,5,3] = 1 / 2 * k * s * sin(k*s)
        
        T[4,0,0] = 1 / 4 * k**2 * (s - cosh(k*s)*sinh(k*s)/k)
        T[4,1,0] = - 1 / 2 * sinh(k*s)**2
        T[4,1,1] = - 1 / 4 * (s + cosh(k*s)*sinh(k*s)/k)
        
        T[4,2,2] = - 1 / 4 * k**2 * (s - cos(k*s)*sin(k*s)/k)
        T[4,3,2] = 1 / 2 * sin(k*s)**2
        T[4,3,3] = - 1 / 4 * (s + cos(k*s)*sin(k*s)/k)
                
    else:
        R = np.array([[1, s, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, s, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, s/Gamma0**2],
                      [0, 0, 0, 0, 0, 1]
                     ])
        T[4,1,1] = - s / 2
        T[4,3,3] = - s / 2
        
    
    return (R, T)
    
def ThinQuad(q):
    R = np.array([[1, 0, 0, 0, 0, 0],
                  [-q, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, q, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]
                 ])
    T = zeros((6,6,6))
    return (R, T)
    
def Bend(rho, theta):
    R = np.array([[cos(theta), rho*sin(theta),0,0,0,rho*(1-cos(theta))],
                  [-sin(theta)/rho, cos(theta), 0,0,0,sin(theta)],
                  [0,0,1,rho*theta,0,0],
                  [0,0,0,1,0,0],
                  [-sin(theta), rho*(cos(theta)-1),0,0,1, rho*(theta/Gamma0**2 - theta + sin(theta))],
                  [0,0,0,0,0,1]
                 ])
    
    T = zeros((6,6,6))
    T[0,0,0] = - sin(theta)**2 / 2 / rho
    T[0,1,0] = sin(theta) * cos(theta)
    T[0,1,1] = rho * cos(theta) * sin(theta/2)**2
    T[0,3,3] = - 1 / 2 * rho * (1 - cos(theta))
    T[0,5,0] = sin(theta)**2
    T[0,5,1] = - rho * sin(theta) * (cos(theta) - 1)
    T[0,5,5] = - 1 / 2 * rho * sin(theta)**2
    
    T[1,1,1] = - sin(theta) / 2
    T[1,3,3] = - sin(theta) / 2
    T[1,5,0] = sin(theta) / rho
    T[1,5,5] = - sin(theta)
    
    T[2,3,0] = sin(theta)
    T[2,3,1] = rho * (1 - cos(theta))
    T[2,5,3] = rho * theta - rho * sin(theta)
    
    T[4,1,1] = - 1 / 2 * rho * sin(theta)
    T[4,3,3] = - 1 / 2 * rho * sin(theta)
    
    T[4,5,1] = rho * (cos(theta) - 1)
    
    return (R, T)


def EntranceEdge(rho, e=0, gap=5.08e-2, h_pole=0, k1=0, fint=0.5,elegant=False):
    h = 1 / rho
    sec_e  = 1 / cos(e)
    sec_e2 = sec_e * sec_e
    sec_e3 = sec_e2 * sec_e
    tan_e  = tan(e)
    tan_e2 = tan_e*tan_e
    phi = fint * h * gap * sec_e * (1 + sin(e)**2)
    
    R = np.array([[1, 0, 0, 0, 0, 0],
                  [h*tan_e, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, -h*tan(e - phi), 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]
                 ])
             
    T = zeros((6,6,6))
    
    T[0,0,0] = - h / 2 * tan_e2
    T[0,2,2] = h / 2 * sec_e2
    T[1,0,0] = h / 2 * h_pole * sec_e3 + k1 * tan_e
    T[1,0,1] = h * tan_e2
    T[1,0,5] = - h * tan_e
    T[1,2,2] = (-k1 + h * h / 2 + h * h * tan_e2) * tan_e - h / 2 * h_pole * sec_e3
    T[1,2,3] = -h * tan_e2
    T[2,0,2] = h * tan_e2
    T[3,0,2] = -h * h_pole * sec_e3 - 2 * k1 * tan_e
    T[3,0,3] = -h * tan_e2
    T[3,1,2] = -h * sec_e2
    if elegant:
        T[3,2,5] = h * tan(e - phi) - h * phi / np.cos(e - phi)**2
    else:
        T[3,2,5] = h * tan_e - h * phi / np.cos(e - phi)**2
    return (R, T)


def ExitEdge(rho, e=0, gap=5.08e-2, h_pole=0, k1=0, fint=0.5,elegant=False):
    h = 1 / rho
    sec_e  = 1 / cos(e)
    sec_e2 = sec_e * sec_e
    sec_e3 = sec_e2 * sec_e
    tan_e  = tan(e)
    tan_e2 = tan_e*tan_e
    phi = fint * h * gap * sec_e * (1 + sin(e)**2)
    
    R = np.array([[1, 0, 0, 0, 0, 0],
                  [h*tan_e, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, -h*tan(e - phi), 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]
                 ]) 
    T = zeros((6,6,6))
    
    T[0,0,0] = h / 2 * tan_e2
    T[0,2,2] = -h / 2 * sec_e2
    T[1,0,0] = h / 2 * h_pole * sec_e3 - (-k1 + h * h / 2 * tan_e2) * tan_e
    T[1,0,1] = -h * tan_e2
    T[1,0,5] = -h * tan_e
    T[1,2,2] = (-k1 - h * h / 2 * tan_e2) * tan_e - h / 2 * h_pole * sec_e3
    T[1,2,3] = h * tan_e2
    T[2,0,2] = -h * tan_e2
    T[3,0,2] = -h * h_pole * sec_e3 + (-k1 + h * h * sec_e2) * tan_e
    T[3,0,3] = h * tan_e2
    T[3,1,2] = h * sec_e2
    if elegant:
        T[3,2,5] = h * tan(e - phi) - h * phi / cos(e - phi)**2
    else:
        T[3,2,5] = h * tan_e - h * phi / cos(e - phi)**2
    return (R, T)
    

def Sben(rho, theta, e1=0, e2=0, gap=5.08e-2, h1=0, h2=0,k1=0, fint=0.5, edge_effects=True, elegant=True):
    """
    363, 463, 464 still doesn't agree with elegant
    because elegant uses different T463
    here the if elegant=False then use T463 from Karl Brown
    if enable elegant=True,
    then first and second order matrix should all agree with SBEN in elegant.
    """
    bend_ent  = EntranceEdge(rho, e1, gap, h1, k1, fint, elegant)
    bend_only = Bend(rho, theta)
    bend_ext  = ExitEdge(rho, e2, gap, h2, k1, fint, elegant)
    
    if not edge_effects:
        return bend_only
    else:
        full_bend = [bend_ent, bend_only, bend_ext]    
        return ListMultiply(full_bend)
    
        
def Sextupole(K2, s):
    R = np.array([[1, s, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, s, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, s/Gamma0**2],
                  [0, 0, 0, 0, 0, 1]
                 ])
                 
    T = zeros((6,6,6))
    
    T[0,0,0] = - K2 * s**2 / 4
    T[0,1,0] = - K2 * s**3 / 6
    T[0,1,1] = - K2 * s**4 / 24
    
    T[0,2,2] = K2 * s**2 / 4
    T[0,3,2] = K2 * s**3 / 6
    T[0,3,3] = K2 * s**4 / 24
    
    T[1,0,0] = - K2 * s    / 2
    T[1,1,0] = - K2 * s**2 / 2
    T[1,1,1] = - K2 * s**3 / 6
    
    T[1,2,2] = K2 * s    / 2
    T[1,3,2] = K2 * s**2 / 2
    T[1,3,3] = K2 * s**3 / 6
    
    T[2,2,0] = K2 * s**2 / 2
    T[2,2,1] = K2 * s**3 / 6
    T[2,3,0] = K2 * s**3 / 6
    T[2,3,1] = K2 * s**4 / 12
    
    
    T[3,2,0] = K2 * s
    T[3,2,1] = K2 * s**2 / 2
    T[3,3,0] = K2 * s**2 / 2
    T[3,3,1] = K2 * s**3 / 3
        
    T[4,1,1] = - s / 2
    T[4,3,3] = - s / 2
    
    return (R, T)


def Sextupole3(K2, length):
    """
    only 3rd order terms
    from elegant, compute_matrices.c
    """ 
    # R = np.identity(6)             
    # T = zeros((6,6,6))
    
    R, T = Sextupole(K2, length)
    
    U = zeros((6,6,6,6))
    
    U[0,0,0,0] = power(K2,2)*power(length,4)/48.0 
    U[0,1,0,0] = power(K2,2)*power(length,5)/48.0 
    U[0,1,1,0] = power(K2,2)*power(length,6)/144.0 
    U[0,1,1,1] = power(K2,2)*power(length,7)/1008.0 
    U[0,2,2,0] = power(K2,2)*power(length,4)/48.0 
    U[0,2,2,1] = -power(K2,2)*power(length,5)/240.0 
    U[0,3,2,0] = power(K2,2)*power(length,5)/40.0 
    U[0,3,2,1] = power(K2,2)*power(length,6)/360.0 
    U[0,3,3,0] = power(K2,2)*power(length,6)/240.0 
    U[0,3,3,1] = power(K2,2)*power(length,7)/1008.0 
    U[0,5,0,0] = K2*power(length,2)/4.0 
    U[0,5,1,0] = K2*power(length,3)/6.0 
    U[0,5,1,1] = K2*power(length,4)/24.0 
    U[0,5,2,2] = -K2*power(length,2)/4.0 
    U[0,5,3,2] = -K2*power(length,3)/6.0 
    U[0,5,3,3] = -K2*power(length,4)/24.0 
    U[1,0,0,0] = power(K2,2)*power(length,3)/12.0 
    U[1,1,0,0] = 5.0*power(K2,2)*power(length,4)/48.0 
    U[1,1,1,0] = power(K2,2)*power(length,5)/24.0 
    U[1,1,1,1] = power(K2,2)*power(length,6)/144.0 
    U[1,2,2,0] = power(K2,2)*power(length,3)/12.0 
    U[1,2,2,1] = -power(K2,2)*power(length,4)/48.0 
    U[1,3,2,0] = power(K2,2)*power(length,4)/8.0 
    U[1,3,2,1] = power(K2,2)*power(length,5)/60.0 
    U[1,3,3,0] = power(K2,2)*power(length,5)/40.0 
    U[1,3,3,1] = power(K2,2)*power(length,6)/144.0 
    U[1,5,0,0] = K2*length/2.0 
    U[1,5,1,0] = K2*power(length,2)/2.0 
    U[1,5,1,1] = K2*power(length,3)/6.0 
    U[1,5,2,2] = -K2*length/2.0 
    U[1,5,3,2] = -K2*power(length,2)/2.0 
    U[1,5,3,3] = -K2*power(length,3)/6.0 
    U[2,2,0,0] = power(K2,2)*power(length,4)/48.0 
    U[2,2,1,0] = power(K2,2)*power(length,5)/40.0 
    U[2,2,1,1] = power(K2,2)*power(length,6)/240.0 
    U[2,2,2,2] = power(K2,2)*power(length,4)/48.0 
    U[2,3,0,0] = -power(K2,2)*power(length,5)/240.0 
    U[2,3,1,0] = power(K2,2)*power(length,6)/360.0 
    U[2,3,1,1] = power(K2,2)*power(length,7)/1008.0 
    U[2,3,2,2] = power(K2,2)*power(length,5)/48.0 
    U[2,3,3,2] = power(K2,2)*power(length,6)/144.0 
    U[2,3,3,3] = power(K2,2)*power(length,7)/1008.0 
    U[2,5,2,0] = -K2*power(length,2)/2.0 
    U[2,5,2,1] = -K2*power(length,3)/6.0 
    U[2,5,3,0] = -K2*power(length,3)/6.0 
    U[2,5,3,1] = -K2*power(length,4)/12.0 
    U[3,2,0,0] = power(K2,2)*power(length,3)/12.0 
    U[3,2,1,0] = power(K2,2)*power(length,4)/8.0 
    U[3,2,1,1] = power(K2,2)*power(length,5)/40.0 
    U[3,2,2,2] = power(K2,2)*power(length,3)/12.0 
    U[3,3,0,0] = -power(K2,2)*power(length,4)/48.0 
    U[3,3,1,0] = power(K2,2)*power(length,5)/60.0 
    U[3,3,1,1] = power(K2,2)*power(length,6)/144.0 
    U[3,3,2,2] = 5.0*power(K2,2)*power(length,4)/48.0 
    U[3,3,3,2] = power(K2,2)*power(length,5)/24.0 
    U[3,3,3,3] = power(K2,2)*power(length,6)/144.0 
    U[3,5,2,0] = -K2*length 
    U[3,5,2,1] = -K2*power(length,2)/2.0 
    U[3,5,3,0] = -K2*power(length,2)/2.0 
    U[3,5,3,1] = -K2*power(length,3)/3.0 
    U[4,1,0,0] = -K2*power(length,2)/4.0 
    U[4,1,1,0] = -K2*power(length,3)/6.0 
    U[4,1,1,1] = -K2*power(length,4)/24.0 
    U[4,2,2,1] = K2*power(length,2)/4.0 
    U[4,3,2,0] = K2*power(length,2)/2.0 
    U[4,3,2,1] = K2*power(length,3)/3.0 
    U[4,3,3,0] = K2*power(length,3)/6.0 
    U[4,3,3,1] = K2*power(length,4)/8.0 
    
    U[4,:,:,:] *= -1
    
    return (R, T, U)
    
def Octupole(K3, s):
    """
    from elegant, compute_matrices.c
    """
    
    R = np.array([[1, s, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, s, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, s/Gamma0**2],
                  [0, 0, 0, 0, 0, 1]
                 ])
                           
    T = zeros((6,6,6))
    
    T[4,1,1] = - s / 2
    T[4,3,3] = - s / 2
    
    U = zeros((6,6,6,6))
    
    L = s
    L2 = L*L
    L3 = L2*L
    L4 = L3*L
    L5 = L4*L

    
    U[0,0,0,0] = -(K3*L2)/12.
    U[0,1,0,0] = -(K3*L3)/36.
    U[0,1,1,0] = -(K3*L4)/72.
    U[0,1,1,1] = -(K3*L5)/120.
    U[0,2,2,0] = (K3*L2)/4.
    U[0,2,2,1] = (K3*L3)/12.
    U[0,3,2,0] = (K3*L3)/12.
    U[0,3,2,1] = (K3*L4)/24.
    U[0,3,3,0] = (K3*L4)/24.
    U[0,3,3,1] = (K3*L5)/40.
    U[1,0,0,0] = -(K3*L)/6.
    U[1,1,0,0] = -(K3*L2)/12.
    U[1,1,1,0] = -(K3*L3)/18.
    U[1,1,1,1] = -(K3*L4)/24.
    U[1,2,2,0] = (K3*L)/2.
    U[1,2,2,1] = (K3*L2)/4.
    U[1,3,2,0] = (K3*L2)/4.
    U[1,3,2,1] = (K3*L3)/6.
    U[1,3,3,0] = (K3*L3)/6.
    U[1,3,3,1] = (K3*L4)/8.
    U[2,2,0,0] = (K3*L2)/4.
    U[2,2,1,0] = (K3*L3)/12.
    U[2,2,1,1] = (K3*L4)/24.
    U[2,2,2,2] = -(K3*L2)/12.
    U[2,3,0,0] = (K3*L3)/12.
    U[2,3,1,0] = (K3*L4)/24.
    U[2,3,1,1] = (K3*L5)/40.
    U[2,3,2,2] = -(K3*L3)/36.
    U[2,3,3,2] = -(K3*L4)/72.
    U[2,3,3,3] = -(K3*L5)/120.
    U[3,2,0,0] = (K3*L)/2.
    U[3,2,1,0] = (K3*L2)/4.
    U[3,2,1,1] = (K3*L3)/6.
    U[3,2,2,2] = -(K3*L)/6.
    U[3,3,0,0] = (K3*L2)/4.
    U[3,3,1,0] = (K3*L3)/6.
    U[3,3,1,1] = (K3*L4)/8.
    U[3,3,2,2] = -(K3*L2)/12.
    U[3,3,3,2] = -(K3*L3)/18.
    U[3,3,3,3] = -(K3*L4)/24.
    
    return R, T, U
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # Transport # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def transport(beam, R, T):        
    assert beam.shape[1] == 6
    assert R.shape == (6,6)

    return np.einsum('ij,...j->...i', R,  beam)

def transport2(beam, R, T):        
    assert beam.shape[1] == 6
    assert R.shape == (6,6)
    assert T.shape == (6,6,6)
    
    return np.einsum('ij,...j->...i', R,  beam) + np.einsum('ijk,...j,...k->...i', T, beam, beam)

def transport3(beam, R, T, U):        
    assert beam.shape[1] == 6
    assert R.shape == (6,6)
    assert T.shape == (6,6,6)
    assert U.shape == (6,6,6,6)
    
    return np.einsum('ij,...j->...i', R,  beam) + \
           np.einsum('ijk,...j,...k->...i', T, beam, beam) + \
           np.einsum('ijkl,...j,...k,...l->...i', U, beam, beam, beam)
    
    
def TMultiply(B, A):
    TMatrix = np.einsum('il,ljk',B[0],A[1]) + np.einsum('ilm,lj,mk',B[1], A[0], A[0])
                
    for i in range(6):
        for j in range(6):
            for k in range(6):
                if j < k:
                    TMatrix[i, j, k] = TMatrix[i, j, k] + TMatrix[i, k, j]
                    TMatrix[i, k, j] = 0
    
    RMatrix = B[0] @ A[0]
    
    return (RMatrix, TMatrix)


def UMultiply(B, A):
    """
    for referece, https://github.com/epicsdeb/elegant/blob/master/apps/src/elegant/concat_mat.c
    """
#    assert len(A) == len(B) == 3
#    RMatrix, TMatrix = TMultiply(B, A)
#    
#    UMatrix = np.einsum('ij,jmnp', B[0], A[2]) + \
#    
#              np.einsum('ijk,jm,knp',B[1], A[0], A[1]) + \
#              np.einsum('ijk,km,jnp',B[1], A[0], A[1]) + \
#              
#              np.einsum('ijk,jm,knp',B[1], A[0], A[1]) + \
#              np.einsum('ijk,km,jnp',B[1], A[0], A[1]) + \
#              
#              np.einsum('ijk,jm,knp',B[1], A[0], A[1]) + \
#              np.einsum('ijk,km,jnp',B[1], A[0], A[1]) + \
#                           
#              np.einsum('ijkl,jm,kn,lp',A[2], B[0], B[0], B[0]) + \
#              np.einsum('ijkl,jm,kp,ln',A[2], B[0], B[0], B[0]) + \
#              np.einsum('ijkl,jn,km,lp',A[2], B[0], B[0], B[0]) + \
#              np.einsum('ijkl,jn,kp,lm',A[2], B[0], B[0], B[0]) + \
#              np.einsum('ijkl,jp,km,ln',A[2], B[0], B[0], B[0]) + \
#              np.einsum('ijkl,jp,kn,lm',A[2], B[0], B[0], B[0]) 
#    
#    for i in range(6):
#        for j in range(6):
#            for k in range(6):
#                for l in range(6):
#                    if not (j<=k and k<=l):
#                        _j, _k, _l = sorted((j,k,l))
#                        UMatrix[i, _j, _k, _l] = UMatrix[i, _j, _k, _l] + UMatrix[i, j, k, l]
#                        UMatrix[i, j, k, l] = 0
#                        
#                        
#    return (RMatrix, TMatrix, UMatrix)
    raise NotImplementedError
    
def ListMultiply(ele_list, order=2):
    if order==2:
        return reduce(TMultiply, ele_list[::-1])
    elif order==3:
        return reduce(UMultiply, ele_list[::-1])
    else:
        raise ValueError

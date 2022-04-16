import numpy as np
import numpy.linalg as la
from scipy import linalg


# define pauli matrices, necessary for writing bloch Hamiltonians
S0 = np.array([[1, 0],[0, 1]])
Sx = np.array([[0, 1],[ 1, 0]])
Sy = np.array([[0, -1j],[1j, 0]])
Sz = np.array([[1, 0],[0, -1]])

S00 = np.kron(S0, S0)
S0x = np.kron(S0, Sx)
S0y = np.kron(S0, Sy)
S0z = np.kron(S0, Sz)
Sx0 = np.kron(Sx, S0)
Sxx = np.kron(Sx, Sx)
Sxy = np.kron(Sx, Sy)
Sxz = np.kron(Sx, Sz)
Sy0 = np.kron(Sy, S0)
Syx = np.kron(Sy, Sx)
Syy = np.kron(Sy, Sy)
Syz = np.kron(Sy, Sz)
Sz0 = np.kron(Sz, S0)
Szx = np.kron(Sz, Sx)
Szy = np.kron(Sz, Sy)
Szz = np.kron(Sz, Sz)

# SU(4)-Matrices,   "generalizations of pauli matrices for 4-band models"
G_1 = Sx0
G_2 = Sz0
G_3 = Syx
G_4 = Syy
G_5 = Syz
G_12 = 1/(2j)*( np.dot(G_1, G_2) - np.dot(G_2, G_1))
G_15 = 1/(2j)*( np.dot(G_1, G_5) - np.dot(G_5, G_1))
G_23 = 1/(2j)*( np.dot(G_2, G_3) - np.dot(G_3, G_2))
G_24 = 1/(2j)*( np.dot(G_2, G_4) - np.dot(G_4, G_2))



def HamKMO_TR_1(k_vec, ham_pars):
  '''
  Time Reversal Invariant KMO Hamiltonian (in momentum space)
  Taken from the paper: https://arxiv.org/pdf/1906.08695.pdf
  
  args:
    k_vec: (k_x, k_y) the point in momentum space
    ham_pars: dict that keeps parameters of hamiltonian
  '''

  t = ham_pars['t']
  t2 = ham_pars['t2']
  t3 = ham_pars['t3']
  l = ham_pars['l']
  kx = k_vec[0]
  ky = k_vec[1]

  H = (  (-t*(1+np.cos(ky)+np.cos(2*kx))-t2*np.cos(kx)) * S0x 
          + (-t*(-np.sin(ky)-np.sin(2*kx))-t2*np.sin(kx)) * S0y
          + (-t3*np.sin(2*kx)) * Szz ) 
  
  # Time reversal conjugate
  H_R = ( -1j*l * ((-0.5 * np.sin(kx) + np.sin(ky)) * 1j * Sxx 
             + (0.5 - 0.5*np.cos(kx) + np.cos(ky)) * 1j * Sxy) )


  Hf =  H + H_R
  return Hf


def HamKMO_TR_2(k_vec, ham_pars):
  '''
  Time Reversal Invariant second KMO Hamiltonian (in momentum space)
  Taken from the paper: https://arxiv.org/pdf/1906.08695.pdf
  
  args:
    k_vec: (k_x, k_y) the point in momentum space
    ham_pars: dict that keeps parameters of hamiltonian
  '''

  E1 = ham_pars['E1'] 
  E2 = ham_pars['E2'] 
  t = ham_pars['t'] 
  t2 = ham_pars['t2'] 
  l = ham_pars['l'] 
  kx = k_vec[0]
  ky = k_vec[1]

  H = ( -E1 * (np.cos(kx) + np.cos(ky)) * (S00 + S0z)
        - E2 * (np.cos(kx) + np.cos(ky)) * (S00 - S0z)
        - 2 * t * (np.cos(kx) - np.cos(ky)) * S0x 
        - t2 * (np.sin(kx)* np.sin(ky)) * S0x ) 
  
  # Mixing part that couples two sets of eigenvalues
  H_mix = -l*(np.sin(kx)*Sy0 + np.sin(ky)*Sx0)

  return H + H_mix


def HamBHZ(k_vec, ham_pars):
  '''
  Time reversal invariant Hamiltonian proosed by Bernevig-Hughes, Zhang (in momentum space)
  Taken from the paper: https://arxiv.org/pdf/cond-mat/0611399
  
  args:
    k_vec: (k_x, k_y) the point in momentum space
    ham_pars: dict that keeps parameters of hamiltonian
  '''
  
  kx = k_vec[0]
  ky = k_vec[1]
  A = ham_pars['A']
  B = ham_pars['B']
  C = ham_pars['C']
  D = ham_pars['D']
  M = ham_pars['M']
  delta = ham_pars['delta']


  E_k = C - 2*D*(2- np.cos(kx) - np.cos(ky))
  
  d_1 = A * np.sin(kx)
  d_2 = A * np.sin(ky) 
  d_3 = -2*B * (2 - M/(2*B) -np.cos(kx) - np.cos(ky))
  
  H = E_k*S00 + d_1*Szx + d_2*S0y + d_3*S0z
  
  # term breaking the inversion symmetry
  H_int = delta*(-Syy)

  return H + H_int


def KaneMele(k_vec, ham_pars):
  '''
  Time reversal invariant Hamiltonian proosed by Kane and Mele (in momentum space)
  Taken from the paper: https://arxiv.org/pdf/cond-mat/0411737
  
  args:
    k_vec: (k_x, k_y) the point in momentum space
    ham_pars: dict that keeps parameters of hamiltonian
  '''
  
  t = ham_pars['t']
  l_v = ham_pars['l_v']
  l_R = ham_pars['l_R']
  l_SO = ham_pars['l_SO']
  kx = k_vec[0]
  ky = k_vec[1]
  

  d_1 = t * (1 + 2*np.cos((kx-ky)/2) * np.cos((kx+ky)/2))
  d_2 = l_v
  d_3 = l_R*(1 - np.cos((kx-ky)/2)*np.cos((kx+ky)/2))
  d_4 = - np.sqrt(3) * l_R * np.sin((kx-ky)/2) *np.sin((kx+ky)/2)
  d_12 = - 2*t*np.cos((kx-ky)/2) * np.sin((kx+ky)/2)
  d_15 = l_SO * (2*np.sin(2*(kx-ky)/2) - 4*np.sin((kx-ky)/2) * np.cos((kx+ky)/2))
  d_23 = -l_R * np.cos((kx-ky)/2) * np.sin((kx+ky)/2)
  d_24 = np.sqrt(3) * l_R * np.sin((kx-ky)/2) * np.cos((kx+ky)/2)

  H =  d_1*G_1 + d_2*G_2 + d_3*G_3 + d_4*G_4 + d_12*G_12 + d_15*G_15 + d_23*G_23 + d_24*G_24

  return H



def uniTrans(old, H):
  '''
  old: set of eigenvectors as a matrix
  H: Hamiltonian whose eigenvectors we will produce. This hamiltonian is very close to the point at which old is calculated
  
  Returns eigenvectors which are contnuation of old at very close neighbor.
  '''
  eival,new=la.eigh(H)
  n = H.shape[0]
  
  if len(set(eival)) != len(eival):
    L = 1j*np.zeros((n,n))
    for k in range(n):
      for j in range(n):
        
        # this condtion corresponds to the unitary transformation only in degenerate subspace isn,
        if eival[j] == eival[k]:             
          L[k][j] = np.vdot(old[:,k], new[:,j])
          
    U, s, Vh = linalg.svd(L)
    G = np.dot(np.transpose(np.conjugate(Vh)), np.transpose(np.conjugate(U)))
    new = np.dot(new, G)
  
  temp = new.copy()
  temp_eval = eival.copy()
  
  # this loop check if there is band inversion and rearranges the bands accordingly
  for u in range(H.shape[0]):                          
    for v in range(H.shape[0]):
      if np.absolute(np.vdot(old[:,u], temp[:,v])) > 0.8:
        new[:,u] = temp[:,v]
        eival[u] = temp_eval[v]
        break
        
  return eival,new



def energyBand(klist, Hamil, ham_pars):
    """
    This function computes the energy band diagram of a given system.
    
    Note that this will be used for any Hamiltonian of any dimension with a proper set of variables and momentum tracjector klist.
    The generality of this code will be very handy.
    
    Parameters:
    -----------
    klist: the path on momentum space along which we draw the energy-momentum plot.
    Hamil: bloch Hamiltonian.
    ham_pars: dict of constants in the bloch Hamiltonian and has to be given in a particular order.
    
    Returns:
    energies eigenvalue and eigenvector along bath of bloch hamiltonian
    """

    for i,k in enumerate(klist):
      H = Hamil(k, ham_pars)
      if i == 0:
        n = H.shape[0]
        evals_mat = np.zeros((len(klist),n))
        evecs_mat = np.empty((len(klist),n),dtype=object)
        evals, ekets = la.eigh(H)
      else:
        evals, ekets = uniTrans(old, H)
        
      for p in range(n):
        evecs_mat[i][p] = ekets[:,p]
        evals_mat[i][p] = evals[p]
      old = ekets
      
      
    return evals_mat,evecs_mat


def berry_phase_Tr_line(k_path, Hamil, var_vec, ns):
    '''
    Calculate the Berry phase of some bands of a Hamiltonian along a certain path.

    Parameters
    ----------
    Hamil : function for bloch hamiltonian
    k_path : The path along which to calculate the Berry phase.
    ns : The sequences of bands whose Berry phases are wanted.

    Returns
    -------
    1d ndarray
        The wanted Berry phase of the selected bands in units of 2 pi.
    '''
    final = np.ones(len(ns))
    for i,k in enumerate(k_path):
      H = Hamil(k, var_vec)
      if i ==0:
        n = H.shape[0]
        evals, ekets = la.eigh(H)
        evs = ekets
        result=np.ones(len(ns), dtype=complex)
      else:
        evals, ekets = uniTrans(old, H)
        for j in ns:
          result[j] *= np.vdot(old[:,j],ekets[:,j])
      old=ekets
    for j in ns:
      result[j] *= np.vdot(old[:,j],evs[:,j])
      
      # in units of 2pi
      final[j] = np.real(np.angle(result[j]))/(2*np.pi)             

    return final


def Z2_invariant(k_path, Hamil, var_vec, ns):
    '''
    Calculate the Berry phase of some bands of a Hamiltonian along a certain path.

    Parameters
    ----------
    Hamil : function for bloch hamiltonian
    k_path : The path along which to calculate the Berry phase.
    ns : The sequences of kramer pairs of bands whose Berry phases are wanted.

    Returns
    -------
    1d ndarray
        The wanted Berry phase of the selected bands in units of 2 pi.
    '''

    for i,k_vec in enumerate(k_path):
      H = Hamil(k_vec, var_vec)
      W = np.zeros((len(ns),len(ns)), dtype=complex)
      if i ==0:
        evals, ekets = la.eigh(H)
        result=np.eye(len(ns), dtype=complex)
        evs = ekets
      else:
        evals, ekets = la.eigh(H)
        for u in ns:
          for v in ns:
            W[u][v] = np.vdot(old[:,u], ekets[:,v])

        result = np.dot(result, W)

      old=ekets

    for u in ns:
      for v in ns:
        W[u][v] = np.vdot(old[:,u], evs[:,v])
    result = np.dot(result, W)
    
    r_list = []
    for i in ns:
        r_list.append(np.angle(la.eig(result)[0][i])/(2*np.pi))

    # taking the eigenvalueso of Wilson loop
    return r_list                              





def k_path(point_arr, loop, N):
  
  '''
  Produces the path connecting the set of points given in the point_arr input.

  Parameters
  ----------
  point_arr : Array of points to form th path
  loop : whether the result is loop or not
  ns : Number of points in resultant path

  Returns
  -------
  2d array (array of 1d array paths)
      Array of paths (each path is a line)
  '''

  k_trajectory = np.empty([0,2])
  
  # Path is not a loop
  for i in range(len(point_arr)-1):
    # Get one line segment
    segment = np.linspace(point_arr[i], point_arr[i+1], N)[1:]
    # Append the segment
    k_trajectory = np.append(k_trajectory, segment, axis=0)
  
  # Path is a loop
  if loop:
    # Get one line segment
    segment = np.linspace(point_arr[-1], point_arr[0], N)[1:]
    # Append the segment
    k_trajectory = np.append(k_trajectory, segment, axis=0)
    
  return k_trajectory
    
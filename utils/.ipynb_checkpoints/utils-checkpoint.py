from more_itertools import consecutive_groups
import numpy as np
from numpy.polynomial import Polynomial

def ang(l, p):
  '''
  Takes a number l, and returns another number located in the interval (-p/2, p/2) using the p-periodicity.
  '''
  x=l
  if np.abs(x)//p !=0:
    g = x % p
    if g > p/2:
      x = g-p
  return x



def middle(x,y, N):
  #returns the average of x and y with periodicity N so that mid1 is right to x, left to y
  #vice versa for for mid2
  if y > x:
    mid1 = (y+x)//2
    mid2 = ((y+x-N)//2)%N
  else:
    mid1 = ((y+x-N)//2)%N
    mid2 = (y+x)//2

  return mid1, mid2
    
  

def maxLenInt(arr, N):
  x = np.nonzero(arr)[0]
  ConList = [list(group) for group in consecutive_groups(x)]
  if ConList[-1][-1] == ConList[0][0] + N-1:
    ConList[0] = ConList[-1] + ConList[0]
    ConList.pop(-1)
  length = len(ConList)
  distArray= np.zeros((length,2), dtype=int)
  for i in range(length):
    x = middle(ConList[i][-1], ConList[(i+1)%length][0], N)[0]
    distArray[i,1] = x 
    distArray[(i+1)%length,0] = x 
  return ConList, distArray


def interval(x,y, L, direction):
  '''
  A periodicty interval function necessary in the main function.
  x: beginning of the inteval
  y: end of the interval
  L: periodicity
  direction: the direction with which the interval proceeds.
  '''
  p=x
  int_list = []
  flag=True
  if direction=="forward":
    while flag==True:
      int_list.append(p)
      if p%L==y:
        flag=False
      p += 1
  elif direction=="backward":
    while flag==True:
      int_list.append(p)
      if p%L ==y:
        flag=False
      p -= 1
  return int_list


def BinarCont(pointList, setPoint, deg, m):
  '''
  pointList: array of numbers to be filled with continuous version of setPoint
  setPoint: arrays whose continuous version will be returned
  deg: degree of polynomial fitting to be used
  m: number of points to be used in the polynomial fitting

  returns a continuous version of setPoint
  '''
  L = setPoint.shape[0]
  step = 2*np.pi/(L)                    #needed for correct periodicity
  u, v = maxLenInt(pointList[:,0], L)

  for elem in range(len(u)):
    # look at the island of determined points
    # go forward
    last_pt1 = []
    last_loc1 = []
    for ind in interval(u[elem][0]%L, v[elem][1], L, "forward"):                 # (u[elem][:-1] + list(range(u[elem][-1], v[elem][1] + 1))) :
      #start to fill up
      q = (ind) % L              #periodicity
      #after this it is as before
      if pointList[q][0] !=0:
        last_pt1.append(pointList[q][0])

      else:                  #if not filled use polynomial fitting
        p = Polynomial.fit(last_loc1,  last_pt1, deg)
        if (setPoint[q][0] - ang(p(ind), 1/step**2))**2 < (setPoint[q][1] - ang(p(ind), 1/step**2))**2:                  #comparison for which function is closer
          pointList[q] = setPoint[q]
          last_pt1.append(setPoint[q][0] + p(ind) - ang(p(ind), 1/step**2))
        else:
          pointList[q][0] = setPoint[q][1]
          pointList[q][1] = setPoint[q][0]
          last_pt1.append(setPoint[q][1]  + p(ind) - ang(p(ind), 1/step**2) )
      last_loc1.append(ind)

      if len(last_pt1) > m:                                                        #take at most m point for pol. fitting
        last_pt1 = last_pt1[1:]
        last_loc1 = last_loc1[1:]

    #go backwards
    last_pt1 = []
    last_loc1 = []
    for ind in interval(u[elem][-1], v[elem][0], L, "backward"):           #(list(reversed(u[elem]))[:-1] + list(reversed(range(v[elem][0], u[elem][0]+1)))):
      #start to fill up
      q = (ind) % L              #periodicity
      #after this it is as before
      if pointList[q][0] !=0:
        last_pt1.insert(0,pointList[q][0])

      else:
        p = Polynomial.fit(last_loc1,  last_pt1, deg)
        if (setPoint[q][0] - ang(p(ind), 1/step**2))**2 < (setPoint[q][1] - ang(p(ind), 1/step**2))**2:
          pointList[q] = setPoint[q] 
          last_pt1.insert(0, setPoint[q][0] + p(ind) - ang(p(ind), 1/step**2))

        else:
          pointList[q][0] = setPoint[q][1]
          pointList[q][1] = setPoint[q][0]
          last_pt1.insert(0,setPoint[q][1]  + p(ind) - ang(p(ind), 1/step**2))

      last_loc1.insert(0,ind)
    
      if len(last_pt1) > m:
        last_pt1 = last_pt1[:-1]
        last_loc1 = last_loc1[:-1]

  return pointList




def cont2D(berr1, berr2, deg, m):

  '''
  Final function to return continuous pair of eigenvalues 
  berr1 and berr2: pair of eigenvalues that are not neccesarily continuous
  return continuous version of these.
  '''

  N = berr1.shape[0]
  #empty array to be filled
  result_berry1 = np.zeros((N,N))
  result_berry2 = np.zeros((N,N))

  #maximum differences of the rows
  maxRow = []
  for i in range(N//2):
    p = max(berr1[i] - berr2[i])
    maxRow.append([p,i])
  maxRow = np.array(maxRow)
  #the element of array with the most gap
  #the process will start from here since the polynomial fitting is the most stable here.
  max_elem = max(maxRow[:,0])
  index = np.where(maxRow[:,0] == max_elem)[0][0]

  for i in range(index, N//2):
    berr_temp = np.zeros((N,2))
    diff_list = berr1[i] - berr2[i]
    max_item = max(diff_list)
    index_max = np.where(diff_list == max_item)[0][0]
    berr_temp[index_max] = berr1[i][index_max], berr2[i][index_max]
    berr_temp[index_max-1] = berr1[i][index_max-1], berr2[i][index_max-1]

    u = BinarCont(berr_temp, np.vstack((berr1[i], berr2[i])).T, deg, m) 

    if i==index:
      result_berry1[i], result_berry2[i] = u[:,0], u[:,1]

    else:
      dist_sum1 = 0
      dist_sum2 = 0
      for k in range(N):
        dist_sum1 += (result_berry1[i-1][k]- u[k][0])**2
        dist_sum2 += (result_berry2[i-1][k]- u[k][0])**2
      if dist_sum1 < dist_sum2:                                         #comparsion of which rows are closer to each other to determine the class of the current row
        result_berry1[i], result_berry2[i] = u[:,0], u[:,1]
      else:
        result_berry1[i], result_berry2[i] = u[:,1], u[:,0]
    result_berry1[N-1-i] = -np.array(list(reversed(result_berry2[i])))
    result_berry2[N-1-i] = -np.array(list(reversed(result_berry1[i])))
  for i in list(reversed(range(index))):
    berr_temp = np.zeros((N,2))
    diff_list = berr1[i] - berr2[i]
    max_item = max(diff_list)
    index_max = np.where(diff_list == max_item)[0][0]

    berr_temp[index_max] = berr1[i][index_max], berr2[i][index_max]
    berr_temp[index_max-1] = berr1[i][index_max-1], berr2[i][index_max-1]

    
    u = BinarCont(berr_temp,  np.vstack((berr1[i], berr2[i])).T, deg, m)

    if i==index:
      result_berry1[i], result_berry2[i] = u[:,0], u[:,1]
    else:
      dist_sum1 = 0
      dist_sum2 = 0
      for k in range(N):
        dist_sum1 += (result_berry1[i+1][k]- u[k][0])**2
        dist_sum2 += (result_berry2[i+1][k]- u[k][0])**2
      if dist_sum1 < dist_sum2:
        result_berry1[i], result_berry2[i] = u[:,0], u[:,1]
      else:
        result_berry1[i], result_berry2[i] = u[:,1], u[:,0]
    result_berry1[N-1-i] = -np.array(list(reversed(result_berry2[i])))
    result_berry2[N-1-i] = -np.array(list(reversed(result_berry1[i])))

  return np.array([result_berry1, result_berry2])


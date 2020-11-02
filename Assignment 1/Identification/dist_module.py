import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
  sq = np.sum(x)
  sv = np.sum(y)
  if sq == 0 or sv == 0:
    return 1
  else:
    sumin = np.sum(list(map(lambda q, v: min(q, v), x, y)))
    return 1 - 0.5*sumin*(sq**(-1)+sv**(-1))



# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    out = np.sum(list(map(lambda q, v: (q-v)**2, x, y)))
    return out



# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
  for i in range(len(x)):
    if x[i] == 0 and y[i] == 0:
      x[i] , y[i] = 1, 1
  out = np.sum(list(map(lambda q, v: (q-v)**2/(q+v), x, y)))
  return out




def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  





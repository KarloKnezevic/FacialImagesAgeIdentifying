#!/usr/bin/env python

import numpy as np

def LDACompute(uzorci, pripadnost):
    # Spoji matricu uzoraka i vektor pripadnosti
    uzorci = np.column_stack([uzorci, pripadnost])

    # Lista svih razreda kojima pripadaju uzorci
    razredi = np.unique(pripadnost)

    # Srednja vrijednost uzoraka (po razredu)
    temp = []
    for razred in razredi:
        temp.append(np.average(uzorci[uzorci[:,-1] == razred], axis=0))
        m = np.array(temp)

    # Srednja vrijendost svih uzoraka
    M = np.average(uzorci, axis=0)

    # Racunanje matrice SW
    dim= uzorci[:,:-1].shape
    SW = np.zeros((dim[1], dim[1]))
    for razred in razredi:
        for uzorak in uzorci[uzorci[:,-1] == razred]:
            temp = (uzorak - m[m[:,-1] == razred])[:,:-1]
            tempTrans = np.transpose(temp)
            SW += temp*tempTrans
  
    # Racunanje matrice SB
    SB = np.zeros((dim[1], dim[1]))
    for razred in razredi:
        n = len(uzorci[uzorci[:,-1] == razred])
        temp = (m[m[:,-1] == razred] - M)[:,:-1]
        tempTrans = np.transpose(temp)
        SB += n*temp*tempTrans
    
    # Racunanje svojstvenih vektora
    lamda, eigenvektori = np.linalg.eig(np.dot(np.linalg.inv(SW),SB))
    
    return eigenvektori[:, 0:3].real
    
def LDAProject(uzorci, eigenvektori):
    rezultat = np.dot(uzorci, eigenvektori)
    return rezultat
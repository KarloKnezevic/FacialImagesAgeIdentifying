#!/usr/bin/env python

import numpy as np

class KNN:
    
    k = 1
    uzorci = None
    pripadnost = None
    
    def train(self, uzorci, pripadnost, k):
        
        self.uzorci = uzorci
        self.pripadnost = pripadnost
        self.k = k
    
    def predict(self, testUzorak):
        
        udaljenosti = []
        for uzorak in self.uzorci:
            d = np.sqrt(np.sum(np.power(testUzorak-uzorak, 2)))
            udaljenosti.append(d)
        udaljenosti = np.column_stack([np.array(udaljenosti), self.pripadnost])
        udaljenosti = udaljenosti[udaljenosti[:,0].argsort()]
        najbolji = udaljenosti[0:self.k, 1]
        najbolji = np.array(najbolji, dtype='int32')
        brojac = np.bincount(najbolji)
        return np.argmax(brojac)
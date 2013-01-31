#!/usr/bin/env python

import os
import re
import lda
import knn
import math
import bisect
import cv2 as cv
import numpy as np

class AgeKlasifikator:
    
    dir_za_ucenje = None
    skupine = [13, 21, 60]
    svmParams = dict(kernel_type = cv.SVM_LINEAR)
    pcaMaxZnacajki = None
    brOkvira = 1
    klasifikator = None
    
    def __init__(self, dir):
        self.dir_za_ucenje = dir
        
    def postaviSkupine(self, skupine):
        self.skupine = skupine

    def postaviSVM(self, params):
        self.svmParams = params
        
    def postaviPCA(self, brZnacajki):
        self.pcaMaxZnacajki = brZnacajki
        
    def postaviBrOkvira(self, brOkvira):
        self.brOkvira = brOkvira

    def ucitajSliku(self, slikaIme, dir, okvirID):
        imgraw = cv.imread(os.path.join(dir, slikaIme), 0)
        velicinaOkvira = (128/int((math.sqrt(self.brOkvira))))
        graniceOkvira = range(0, 129, velicinaOkvira)
        br = 0
        for i, row in enumerate(graniceOkvira[:-1]):
            for j, col in enumerate(graniceOkvira[:-1]):
                if br == okvirID:
                    imgraw = imgraw[row:graniceOkvira[i+1], col:graniceOkvira[j+1]]
                br+=1
        return imgraw.reshape(velicinaOkvira*velicinaOkvira)
    
    def dobaviRazred(self, slikaIme):
        dob = int(slikaIme.split(".")[0][6:8])
        return (bisect.bisect(self.skupine, dob), dob)

    def doPCALDA(self):
        if self.brOkvira in (1, 4, 16):
            
            self.allEigenvektoriLDA = []
            self.allEigenvektoriPCA = []
            self.allSrednjaVrij = []
            self.ldaMatricaUzorakaFinal = None
            
            for okvirID in range(self.brOkvira):
                
                ''' Ucitavanje slika u matricu '''
                matricaUzoraka = None
                self.pripadnost = np.array([])
                self.dob = np.array([])
                for slikaIme in os.listdir(self.dir_za_ucenje):
                    imgvektor = self.ucitajSliku(slikaIme, self.dir_za_ucenje, okvirID)
                    pripadnost, dob = self.dobaviRazred(slikaIme)
                    self.pripadnost = np.append(self.pripadnost, pripadnost)
                    self.dob = np.append(self.dob, dob)
                    try:
                        matricaUzoraka = np.vstack((matricaUzoraka, imgvektor))
                    except:
                        matricaUzoraka = imgvektor
                
                ''' Racunanje PCA '''
                srednjaVrij = np.mean(matricaUzoraka, axis=0).reshape(1,-1)
                if self.pcaMaxZnacajki == None:
                    self.pcaMaxZnacajki = matricaUzoraka.shape[0]-4

                srednjaVrij, eigenvektoriPCA = cv.PCACompute(matricaUzoraka, srednjaVrij, maxComponents=self.pcaMaxZnacajki)
                self.allSrednjaVrij.append(srednjaVrij)
                self.allEigenvektoriPCA.append(eigenvektoriPCA)
                pcaMatricaUzoraka = cv.PCAProject(matricaUzoraka, srednjaVrij, eigenvektoriPCA)
                
                ''' Racunanje LDA '''
                eigenvektoriLDA = lda.LDACompute(pcaMatricaUzoraka, self.pripadnost)
                self.allEigenvektoriLDA.append(eigenvektoriLDA)
                ldaMatricaUzoraka = lda.LDAProject(pcaMatricaUzoraka, eigenvektoriLDA)
                
                ''' Sastavljanje finalne LDA matrice znacajki'''
                try:
                    self.ldaMatricaUzorakaFinal = np.column_stack([self.ldaMatricaUzorakaFinal, ldaMatricaUzoraka])
                except:
                    self.ldaMatricaUzorakaFinal = ldaMatricaUzoraka
                
                if self.brOkvira > 1:
                    print "Ucitavanje te racunanje PCA i LDA za okvir s ID-om %s ZAVRSENO" % str(okvirID+1)
                else:
                    print "Ucitavanje slika te racunanje PCA i LDA ZAVRSENO"
        
        else:
            print "Nedopustena vrijednost za broj okvira! Dopusteno je 1, 4, 16!"
    
    def trainSVM(self):
        ''' Treniranje SVM '''
        ldaMatricaUzorakaFinal = np.array(self.ldaMatricaUzorakaFinal, dtype='float32')
        pripadnost = np.array(self.pripadnost, dtype='float32')
        self.klasifikator = cv.SVM()
        self.klasifikator.train(ldaMatricaUzorakaFinal, pripadnost, params = self.svmParams)
            
    def trainKNN(self, k):
        ''' Treniranje KNN '''
        self.klasifikator = knn.KNN()
        self.klasifikator.train(self.ldaMatricaUzorakaFinal, self.pripadnost, k)
    
    def predict(self, slikaIme, dirZaTestiranje):
        ''' Predvidanje za jednu sliku '''
        ldaImgVektorFinal = None
        if self.brOkvira in (1, 4, 16):
            for okvirID in range(self.brOkvira):
                imgvektor = np.array([self.ucitajSliku(slikaIme, dirZaTestiranje, okvirID)])
                pcaImgvektor = cv.PCAProject(imgvektor, self.allSrednjaVrij[okvirID], self.allEigenvektoriPCA[okvirID])
                ldaImgvektor = lda.LDAProject(pcaImgvektor, self.allEigenvektoriLDA[okvirID])
                try:
                    ldaImgvektorFinal = np.column_stack([ldaImgvektorFinal, ldaImgvektor])
                except:
                    ldaImgvektorFinal = ldaImgvektor
                
            ldaImgvektorFinal = np.array(ldaImgvektorFinal, dtype='float32')
            return self.klasifikator.predict(ldaImgvektorFinal)
        else:
            return -1

    def batchPredict(self, dirZaTestiranje):
        ''' Predvidanje za sve slike u direktoriju '''
        broj_slika = 0
        ispravno_klasificiranih = 0
        self.razlikaRazred = []
        for slikaIme in os.listdir(dirZaTestiranje):
            broj_slika += 1
            prediction = self.predict(slikaIme, dirZaTestiranje)
            pripadnost, dob = self.dobaviRazred(slikaIme)
            self.razlikaRazred.append((pripadnost, prediction))
            if pripadnost == prediction:
                ispravno_klasificiranih += 1
        if broj_slika == 0: return 0
        else:
            return int((float(ispravno_klasificiranih)/broj_slika)*100)

    def ldaTest(self, dirZaTestiranje):
        ''' Koristeno za generiranje 3D grafova LDA rasprsenosti '''
        ldaTestMatrica = None
        pripadnostList = []
        dobList = []
        for slikaIme in os.listdir(dirZaTestiranje):
            imgvektor = np.array([self.ucitajSliku(slikaIme, dirZaTestiranje, 0)])
            pripadnost, dob = self.dobaviRazred(slikaIme)
            pripadnostList.append(pripadnost)
            dobList.append(dob)
            pcaImgvektor = cv.PCAProject(imgvektor, self.allSrednjaVrij[0], self.allEigenvektoriPCA[0])
            ldaImgvektor = lda.LDAProject(pcaImgvektor, self.allEigenvektoriLDA[0])
            try:
                ldaTestMatrica = np.vstack((ldaTestMatrica, ldaImgvektor))
            except:
                ldaTestMatrica = ldaImgvektor
        return (ldaTestMatrica, np.array(pripadnostList), np.array(dobList))
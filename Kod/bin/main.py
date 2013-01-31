#!/usr/bin/env python

import sys
import cv2 as cv
from klasifikator import AgeKlasifikator

''' POSTAVKE '''
DIR_ZA_UCENJE = 'uzorci_za_ucenje/'
DIR_ZA_TESTIRANJE = 'uzorci_za_testiranje/'

''' POCETAK '''
print "Raspoznavanje dobne skupine pomocu linearne diskriminantne analize (LDA)"
print "Odabrani klasifikator ce biti pokrenut s optimalnim parametrima"
izbor = raw_input('Koji klasifikator zelite koristiti? kNN(1), SVM(2) ili EXIT(0): ')

try: izbor = int(izbor)
except:
    print "Pogreska u odabiru. Ponovo pokrenite program!"
    sys.exit()

primjer = AgeKlasifikator(DIR_ZA_UCENJE)
primjer.postaviSkupine([13, 21, 60])

if izbor == 1:
    primjer.postaviBrOkvira(4)
    primjer.postaviPCA(100)
    primjer.doPCALDA()
    primjer.trainKNN(41)
    print "Ispravno (kNN): %s posto!" % str(primjer.batchPredict(DIR_ZA_TESTIRANJE))

elif izbor == 2:
    primjer.postaviBrOkvira(1)
    primjer.postaviPCA(150)
    primjer.doPCALDA()
    primjer.postaviSVM(dict(kernel_type = cv.SVM_POLY, svm_type = cv.SVM_C_SVC, C = 2, degree = 5, gamma = 14))
    primjer.trainSVM()
    print "Ispravno (SVM): %s posto!" % str(primjer.batchPredict(DIR_ZA_TESTIRANJE))
    
elif izbor == 0:
    sys.exit()

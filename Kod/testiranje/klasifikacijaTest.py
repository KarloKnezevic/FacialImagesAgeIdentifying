#!/usr/bin/env python

import cv2 as cv
from klasifikator import AgeKlasifikator

''' POSTAVKE '''
DIR_ZA_UCENJE = 'uzorci_za_ucenje/'
DIR_ZA_TESTIRANJE = 'uzorci_za_testiranje/'

''' FORMAT ZAPISA: ISPRAVNO|KLASIFICIRANO'''

primjer = AgeKlasifikator(DIR_ZA_UCENJE)
primjer.postaviSkupine([13, 21, 60])
primjer.postaviPCA(100)
primjer.doPCALDA()

datoteka = open('SVM_POLY_klasifikacija_testiranje.txt','a')
primjer.postaviSVM(dict(kernel_type = cv.SVM_POLY, svm_type = cv.SVM_C_SVC, C = 1, degree = 1, gamma = 1))
primjer.trainSVM()
primjer.batchPredict(DIR_ZA_TESTIRANJE)
for pripadnost, prediction in primjer.razlikaRazred:
    datoteka.writelines(str(pripadnost) + "|" + str(int(prediction)) + "\n")
datoteka.close()

datoteka = open('SVM_RBF_klasifikacija_testiranje.txt','a')
primjer.postaviSVM(dict(kernel_type = cv.SVM_RBF, svm_type = cv.SVM_C_SVC, C = 1,  gamma = 1))
primjer.trainSVM()
primjer.batchPredict(DIR_ZA_TESTIRANJE)
for pripadnost, prediction in primjer.razlikaRazred:
    datoteka.writelines(str(pripadnost) + "|" + str(int(prediction)) + "\n")
datoteka.close()

datoteka = open('KNN_klasifikacija_testiranje.txt','a')
primjer.trainKNN(1)
primjer.batchPredict(DIR_ZA_TESTIRANJE)
for pripadnost, prediction in primjer.razlikaRazred:
    datoteka.writelines(str(pripadnost) + "|" + str(int(prediction)) + "\n")
datoteka.close()

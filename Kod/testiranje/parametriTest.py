#!/usr/bin/env python

import cv2 as cv
from klasifikator import AgeKlasifikator

''' POSTAVKE '''
DIR_ZA_UCENJE = 'uzorci_za_ucenje/'
DIR_ZA_TESTIRANJE = 'uzorci_za_testiranje/'

primjer = AgeKlasifikator(DIR_ZA_UCENJE)
primjer.postaviSkupine([13, 21, 60])
primjer.postaviPCA(100)
primjer.doPCALDA()

dosadNajbolji = 0
datoteka = open('SVM_POLY_testiranje.txt','a')
print "SVM_POLY"
for i in range(1,10):
    for j in range(1,100,5):
        for k in range(1,40,2):
            print "C=%s, degree=%s, gamma=%s" % (j,i,k)
            primjer.postaviSVM(dict(kernel_type = cv.SVM_POLY, svm_type = cv.SVM_C_SVC, C = j, degree = i, gamma = k))
            primjer.trainSVM()
            rezultat = primjer.batchPredict(DIR_ZA_TESTIRANJE)
            datoteka.writelines(str(rezultat) + "|" + str(j) + "|" + str(i) + "|" + str(k) +"\n")
            if rezultat > dosadNajbolji:
                best = str(rezultat) + "|" + str(j) + "|" + str(i) + "|" + str(k) +"\n"
                dosadNajbolji = rezultat
datoteka.writelines(best)
datoteka.close()

dosadNajbolji = 0
datoteka = open('SVM_RBF_testiranje.txt','a')      
print "SVM_RBF"
for j in range(1,100,5):
    for k in range(1,40,2):
        print "C=%s, gamma=%s" % (j,k)
        primjer.postaviSVM(dict(kernel_type = cv.SVM_RBF, svm_type = cv.SVM_C_SVC, C = j, gamma = k))
        primjer.trainSVM()
        rezultat = primjer.batchPredict(DIR_ZA_TESTIRANJE)
        datoteka.writelines(str(rezultat) + "|" + str(j) + "|" + str(k) +"\n")
        if rezultat > dosadNajbolji:
            best = str(rezultat) + "|" + str(j) + "|" + str(k) +"\n"
            dosadNajbolji = rezultat
datoteka.writelines(best)
datoteka.close()

dosadNajbolji = 0
datoteka = open('KNN_testiranje.txt','a')   
print "KNN"
for k in range(1,10):
    print "K=%s" % k
    primjer.trainKNN(k)
    rezultat = primjer.batchPredict(DIR_ZA_TESTIRANJE)
    datoteka.writelines(str(rezultat) + "|" + str(k) +"\n")
    if rezultat > dosadNajbolji:
        best = str(rezultat) + "|" + str(k) +"\n"
        dosadNajbolji = rezultat
datoteka.writelines(best)
datoteka.close()

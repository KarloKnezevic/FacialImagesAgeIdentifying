#!/usr/bin/env python

from matplotlib import pyplot as Pyplot
from mpl_toolkits.mplot3d import Axes3D
from klasifikator import AgeKlasifikator

''' POSTAVKE '''
DIR_ZA_UCENJE = 'uzorci_za_ucenje/'
DIR_ZA_TESTIRANJE = 'uzorci_za_testiranje/'

# 3D ispis tocaka nakon LDA (radi samo za cijelu sliku...znaci broj okvira 1!!)
test1 = AgeKlasifikator(DIR_ZA_UCENJE)
test1.postaviSkupine([13, 21, 60])
test1.postaviPCA(100)
test1.doPCALDA()

# Uzorci za ucenje
tocke = test1.ldaMatricaUzorakaFinal
pripadnost = test1.pripadnost
dob = test1.dob

# Uzorci za testiranje
ttocke, tpripadnost, tdob = test1.ldaTest(DIR_ZA_TESTIRANJE)

figure = Pyplot.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
ax1 = figure.add_subplot(121, projection='3d')
ax1.scatter(tocke[:,0], tocke[:,1], tocke[:,2], c=pripadnost)
ax2 = figure.add_subplot(122, projection='3d')
ax2.scatter(ttocke[:,0], ttocke[:,1], ttocke[:,2], c=tpripadnost)
Pyplot.savefig("uzorciUcenjeTestRazredi.png")

figure = Pyplot.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
ax1 = figure.add_subplot(121, projection='3d')
ax1.scatter(tocke[:,0], tocke[:,1], tocke[:,2], c=dob)
ax2 = figure.add_subplot(122, projection='3d')
ax2.scatter(ttocke[:,0], ttocke[:,1], ttocke[:,2], c=tdob)
Pyplot.savefig("uzorciUcenjeTestDobi.png")

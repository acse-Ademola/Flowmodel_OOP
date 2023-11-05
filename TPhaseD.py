import os
import sys
import warnings
from math import pi
from time import time
import numpy as np
import pandas as pd
from sortedcontainers import SortedList
from utilities import Computations

class TwoPhaseDrainage():
    def __init__(self, obj):
        global do, _cornArea, _cornCond, _centerArea, _centerCond
        self.obj = obj
        do = Computations(self)

        self.fluid = np.zeros(self.totElements, dtype='int')
        self.fluid[[-1, 0]] = 1   # already filled
        self.trapped = np.zeros(self.totElements, dtype='bool')
        self.contactAng, self.thetaRecAng, self.thetaAdvAng =\
            do.__wettabilityDistribution__()
        self.Fd_Tr = do.__computeFd__(self.elemTriangle, self.halfAngles)
        self.Fd_Sq = do.__computeFd__(
            self.elemSquare, np.array([pi/4, pi/4, pi/4, pi/4]))

        self.cornExistsTr = np.zeros([self.nTriangles, 3], dtype='bool')
        self.cornExistsSq = np.zeros([self.nSquares, 4], dtype='bool')
        self.initedTr = np.zeros([self.nTriangles, 3], dtype='bool')
        self.initedSq = np.zeros([self.nSquares, 4], dtype='bool')
        self.initOrMaxPcHistTr = np.zeros([self.nTriangles, 3])
        self.initOrMaxPcHistSq = np.zeros([self.nSquares, 4])
        self.initOrMinApexDistHistTr = np.zeros([self.nTriangles, 3])
        self.initOrMinApexDistHistSq = np.zeros([self.nSquares, 4])
        self.initedApexDistTr = np.zeros([self.nTriangles, 3])
        self.initedApexDistSq = np.zeros([self.nSquares, 4])
        self.advPcTr = np.zeros([self.nTriangles, 3])
        self.advPcSq = np.zeros([self.nSquares, 4])
        self.recPcTr = np.zeros([self.nTriangles, 3])
        self.recPcSq = np.zeros([self.nSquares, 4])
        self.hingAngTr = np.zeros([self.nTriangles, 3])
        self.hingAngSq = np.zeros([self.nSquares, 4])
        
        self.__computePistonPc__()
        self.PcD = self.PistonPc.copy()
        self.centreEPOilInj = np.zeros(self.totElements)
        self.centreEPOilInj[self.elementLists] = 2*self.sigma*np.cos(
            self.thetaRecAng[self.elementLists])/self.Rarray[self.elementLists]
        
        self.ElemToFill = SortedList(key=lambda i: self.LookupList(i))
        ElemToFill = self.nPores+self.conTToIn
        self.ElemToFill.update(ElemToFill)
        self.NinElemList = np.ones(self.totElements, dtype='bool')
        self.NinElemList[ElemToFill] = False

        _cornArea = self.AreaSPhase.copy()
        _centerArea = np.zeros(self.totElements) 
        _cornCond = self.gwSPhase.copy()
        _centerCond = np.zeros(self.totElements)
        
        #from IPython import embed; embed() 
    @property
    def AreaWPhase(self):
        return _cornArea
    
    @property
    def AreaNWPhase(self):
        return _centerArea
    
    @property
    def gWPhase(self):
        return _cornCond
    
    @property
    def gNWPhase(self):
        return _centerCond


    def __getattr__(self, name):
        return getattr(self.obj, name)
    
    def LookupList(self, k):
        return (self.PcD[k], k > self.nPores, -k)
    
    def drainage(self):
        global capPres
        self.is_oil_inj = True
        start = time()
        print('--------------------------------------------------------------')
        print('---------------------Two Phase Drainage Process---------------')

        self.__writeHeaders__()
        #from IPython import embed; embed()

        self.SwTarget = max(self.finalSat, self.satW-self.dSw*0.5)
        self.capPresMax = 0
        self.PcTarget = min(self.maxPc, self.capPresMax+(
            self.minDeltaPc+abs(
             self.capPresMax)*self.deltaPcFraction)*0.1)
        self.oldPcTarget = 0

        capPres = self.PcD[self.ElemToFill[0]]

        while self.filling:
            self.oldSatW = self.satW
            self.__PDrainage__()        
            
            if (self.PcTarget > self.maxPc-0.001) or (
                 self.satW < self.finalSat+0.00001):
                self.filling = False
                break
            
            self.oldPcTarget = self.capPresMax
            self.PcTarget = min(self.maxPc+1e-7, self.PcTarget+(
                    self.minDeltaPc+abs(
                     self.PcTarget)*self.deltaPcFraction))
            self.SwTarget = max(self.finalSat-1e-15, round((
                    self.satW-self.dSw*0.75)/self.dSw)*self.dSw)

            if len(self.ElemToFill) == 0:
                self.filling = False

                while self.PcTarget < self.maxPc-0.001:
                    self.__CondTP_Drainage__()
                    self.satW = do.Saturation(self.AreaWPhase, self.AreaSPhase)
                    self.SwTarget = self.satW
                    gwL = do.computegL(self.gwP, self.gwT)
                    self.qW = do.computeFlowrate(gwL)
                    self.krw = self.qW/self.qwSPhase
                    if any(self.fluid[self.tList[self.isOnOutletBdr[self.tList]]] == 1):
                        gnwL = do.computegL(self.gNWPhase)
                        self.qNW = do.computeFlowrate(gnwL, phase=1)
                        self.krnw = self.qNW/self.qnwSPhase
                    else:
                        self.qNW, self.krnw = 0, 0

                    self.writeResult(self.fQ1)
                    self.PcTarget = min(self.maxPc+1e-7, self.PcTarget+(
                        self.minDeltaPc+abs(
                         self.PcTarget)*self.deltaPcFraction))
                
                break

        self.rpd = self.sigma/self.PcTarget
        print("Number of trapped elements: ", self.trapped.sum())
        #val, count = np.unique(self.clusterP, return_counts=True)
        #val, count = np.unique(self.clusterT, return_counts=True)
        print(self.rpd, self.sigma, self.PcTarget)
        print('Time spent for the drainage process: ', time() - start)        
        print('==========================================================\n\n')


    def popUpdateOilInj(self):
        global capPres, cntP, cntT
        k = self.ElemToFill.pop(0)
        capPres = self.PcD[k]
        self.capPresMax = np.max([self.capPresMax, capPres])

        try:
            assert k > self.nPores
            ElemInd = k-self.nPores
            assert not do.isTrapped(k, 0)
            self.fluid[k] = 1
            self.PistonPc[k] = self.centreEPOilInj[k]
            ppp = np.array([self.P1array[ElemInd-1], self.P2array[
                ElemInd-1]])
            p = ppp[(self.fluid[ppp] == 0) & ~(self.trapped[ppp])]

            self.__update_PcD_ToFill__(p)
            cntT += 1
            self.invInsideBox += self.isinsideBox[k]
        except AssertionError:
            pass
        except IndexError:
            cntT += 1
            self.invInsideBox += self.isinsideBox[k]

        try:
            assert k <= self.nPores
            assert not do.isTrapped(k, 0)
            self.fluid[k] = 1
            self.PistonPc[k] = self.centreEPOilInj[k]
            thr = self.PTConData[k]+self.nPores
            thr = thr[(self.fluid[thr] == 0) & ~(self.trapped[thr])]

            self.__update_PcD_ToFill__(thr)
            cntP += 1
            self.invInsideBox += self.isinsideBox[k]
        except AssertionError:
            pass
        except IndexError:
            cntP += 1
            self.invInsideBox += self.isinsideBox[k]

    def __PDrainage__(self):
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        global capPres, cntT, cntP
        self.totNumFill = 0
        self.fillTarget = max(self.m_minNumFillings, int(
            self.m_initStepSize*(self.totElements)*(
             self.SwTarget-self.satW)))
        self.invInsideBox = 0

        while (self.PcTarget+1.0e-32 > self.capPresMax) & (
                self.satW > self.SwTarget):
            self.oldSatW = self.satW
            self.invInsideBox = 0
            cntT, cntP = 0, 0
            while (self.invInsideBox < self.fillTarget) & (
                len(self.ElemToFill) != 0) & (
                    self.PcD[self.ElemToFill[0]] <= self.PcTarget):
                self.popUpdateOilInj()
            try:
                assert (self.PcD[self.ElemToFill[0]] > self.PcTarget) & (
                        self.capPresMax < self.PcTarget)
                self.capPresMax = self.PcTarget
            except AssertionError:
                pass
            
            self.__CondTP_Drainage__()
            self.satW = do.Saturation(self.AreaWPhase, self.AreaSPhase)
            self.totNumFill += (cntP+cntT)
            try:
                self.fillTarget = max(self.m_minNumFillings, int(min(
                    self.fillTarget*self.m_maxFillIncrease,
                    self.m_extrapCutBack*(self.invInsideBox / (
                        self.satW-self.oldSatW))*(self.SwTarget-self.satW))))
            except OverflowError:
                pass
                
            try:
                assert self.PcD[self.ElemToFill[0]] <= self.PcTarget
            except AssertionError:
                break

        try:
            assert (self.PcD[self.ElemToFill[0]] > self.PcTarget)
            self.capPresMax = self.PcTarget
        except AssertionError:
            self.PcTarget = self.capPresMax
        
        self.__CondTP_Drainage__()
        self.satW = do.Saturation(self.AreaWPhase, self.AreaSPhase)
        gwL = do.computegL(self.gWPhase)
        self.qW = do.computeFlowrate(gwL)
        self.krw = self.qW/self.qwSPhase
        

        try:
            assert self.fluid[self.tList[self.isOnOutletBdr[self.tList]]].sum() > 0
            gnwL = do.computegL(self.gNWPhase)
            self.qNW = do.computeFlowrate(gnwL, phase=1)
            self.krnw = self.qNW/self.qnwSPhase
        except AssertionError:
            self.qNW, self.krnw = 0, 0

        do.writeResult(self.fQ1)

    
    def __computePc__(self, arrr, Fd): 
        Pc = self.sigma*(1+2*np.sqrt(pi*self.Garray[arrr]))*np.cos(
            self.contactAng[arrr])*Fd/self.Rarray[arrr]
        return Pc
    
    def __computePistonPc__(self) -> None:
        self.PistonPc = np.zeros(self.totElements)
        self.PistonPc[self.elemCircle] = 2*self.sigma*np.cos(
            self.contactAng[self.elemCircle])/self.Rarray[self.elemCircle]
        self.PistonPc[self.elemTriangle] = self.__computePc__(
            self.elemTriangle, self.Fd_Tr)
        self.PistonPc[self.elemSquare] = self.__computePc__(
            self.elemSquare, self.Fd_Sq)
        
    def __func1(self, arr):
        try:
            return self.PistonPc[arr[self.fluid[arr] == 1]].min()
        except ValueError:
            return 0
    
    def __func2(self, i):
        try:
            return self.PistonPc[i[(i > 0) & (self.fluid[i] == 1)]].min()
        except ValueError:
            return 0
    
    def __func3(self, i):
        try:
            self.ElemToFill.remove(i)
        except ValueError:
            pass

    def __update_PcD_ToFill__(self, arr) -> None:
        arrP = arr[arr <= self.nPores]
        arrT = arr[arr > self.nPores]
        try:
            thr = self.PTConData[arrP]+self.nPores
            minNeiPc = np.array([*map(lambda arr: self.__func1(arr), thr)])
            #from IPython import embed; embed()
            entryPc = np.maximum(0.999*minNeiPc+0.001*self.PistonPc[
                arrP], self.PistonPc[arrP])
            
            cond1 = self.NinElemList[arrP]
            cond2 = ~cond1 & (entryPc != self.PcD[arrP])
            try:
                assert cond1.sum() > 0
                self.PcD[arrP[cond1]] = entryPc[cond1]
                self.ElemToFill.update(arrP[cond1])
                self.NinElemList[arrP[cond1]] = False
            except AssertionError:
                pass
            try:
                assert cond2.sum() > 0
                #[self.ElemToFill.remove(p) for p in arrP[cond2]]
                [*map(lambda i: self.__func3(i), arrP[cond2])]
                self.PcD[arrP[cond2]] = entryPc[cond2]
                self.ElemToFill.update(arrP[cond2])
            except AssertionError:
                pass
        except IndexError:
            pass
        try:
            ppp = np.array([*zip(
                self.P1array[arrT-self.nPores-1], self.P2array[arrT-self.nPores-1])])
            minNeiPc = np.array([*map(lambda arr: self.__func2(arr), ppp)])
            entryPc = np.maximum(0.999*minNeiPc+0.001*self.PistonPc[arrT], self.PistonPc[arrT])
            
            cond1 = self.NinElemList[arrT]
            cond2 = ~cond1 & (entryPc != self.PcD[arrT])
            try:
                assert cond1.sum() > 0
                self.PcD[arrT[cond1]] = entryPc[cond1]
                self.ElemToFill.update(arrT[cond1])
                self.NinElemList[arrT[cond1]] = False
            except AssertionError:
                pass
            try:
                assert cond2.sum() > 0
                [*map(lambda i: self.__func3(i), arrT[cond2])]
                #[self.ElemToFill.remove(t) for t in arrT[cond2]]
                self.PcD[arrT[cond2]] = entryPc[cond2]
                self.ElemToFill.update(arrT[cond2])
            except AssertionError:
                pass          
        except IndexError:
            pass

    
    def __CondTP_Drainage__(self):
        # to suppress the FutureWarning and SettingWithCopyWarning respectively
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pd.options.mode.chained_assignment = None
        
        arrr = ((self.fluid == 1) & (~self.trapped))
        arrrS = arrr[self.elemSquare]
        arrrT = arrr[self.elemTriangle]
        arrrC = arrr[self.elemCircle]
        
        # create films
        try:
            assert (arrrT.sum() > 0)
            Pc = self.PcD[self.elemTriangle]
            curConAng = self.contactAng.copy()
            do.createFilms(self.elemTriangle, arrrT, self.halfAngles, Pc,
                        self.cornExistsTr, self.initedTr,
                        self.initOrMaxPcHistTr,
                        self.initOrMinApexDistHistTr, self.advPcTr,
                        self.recPcTr, self.initedApexDistTr)
            
            apexDist = np.zeros(self.hingAngTr.shape)
            conAngPT, apexDistPT = do.cornerApex(
                self.elemTriangle, arrrT, self.halfAngles, self.capPresMax,
                curConAng, self.cornExistsTr, self.initOrMaxPcHistTr,
                self.initOrMinApexDistHistTr, self.advPcTr,
                self.recPcTr, apexDist, self.initedApexDistTr,
                self.hingAngTr)
            
            _cornArea[self.elemTriangle[arrrT]], _cornCond[
                self.elemTriangle[arrrT]] = do.calcAreaW(
                arrrT, self.halfAngles, conAngPT, self.cornExistsTr, apexDistPT) 
        except AssertionError:
            pass

        try:
            assert (arrrS.sum() > 0)
            Pc = self.PcD[self.elemSquare]
            curConAng = self.contactAng.copy()
            do.createFilms(self.elemSquare, arrrS, np.array([pi/4, pi/4, pi/4, pi/4]),
                           Pc, self.cornExistsSq, self.initedSq, self.initOrMaxPcHistSq,
                           self.initOrMinApexDistHistSq, self.advPcSq,
                           self.recPcSq, self.initedApexDistSq)

            apexDist = np.zeros(self.hingAngSq.shape)
            conAngPS, apexDistPS = do.cornerApex(
                self.elemSquare, arrrS, np.array([pi/4, pi/4, pi/4, pi/4]), self.capPresMax,
                curConAng, self.cornExistsSq, self.initOrMaxPcHistSq,
                self.initOrMinApexDistHistSq, self.advPcSq,
                self.recPcSq, apexDist, self.initedApexDistSq,
                self.hingAngSq)
            
            _cornArea[self.elemSquare[arrrS]], _cornCond[
                self.elemSquare[arrrS]] = do.calcAreaW(
                arrrS, np.array([pi/4, pi/4, pi/4, pi/4]), conAngPS, self.cornExistsSq, apexDistPS)
        except AssertionError:
            pass
        try:
            assert (arrrC.sum() > 0)
            _cornArea[self.elemCircle[arrrC]] = 0.0
            _cornCond[self.elemCircle[arrrC]] = 0.0
        except  AssertionError:
            pass
        
        try:
            assert (_cornArea[arrr] <= self.AreaSPhase[arrr]).all()
            assert (_cornCond[arrr] <= self.gwSPhase[arrr]).all()

            _centerArea[arrr] = self.AreaSPhase[arrr] - _cornArea[arrr]
            _centerCond[arrr] = np.where(self.AreaSPhase[arrr] != 0.0, _centerArea[
                arrr]/self.AreaSPhase[arrr]*self.gnwSPhase[arrr], 0.0)
        except AssertionError:
            print('higher values than expected!')
            from IPython import embed; embed()

    
    def __writeHeaders__(self):
        self._num = 1
        while True:
            file_name = os.path.join(self.dirname, "Results/Flowmodel_"+
                                 self.title+"_Drainage_"+str(self._num)+".csv")
            if os.path.isfile(file_name): self._num += 1
            else:
                self.fQ1 = open(file_name, 'a+')
                break
        
        self.fQ1.write('\n======================================================================')
        self.fQ1.write('\n'+'%'+'Fluid properties:\nsigma (mN/m)  \tmu_w (cP)  \tmu_nw (cP)')
        self.fQ1.write('\n'+'%'+ '\t%.6g\t\t%.6g\t\t%.6g' % (
            self.sigma, self.muw, self.munw, ))
        self.fQ1.write('\n'+'%'+' calcBox: \t %.6g \t %.6g' % (
            self.calcBox[0], self.calcBox[1], ))
        self.fQ1.write('\n'+'%'+'Wettability:')
        self.fQ1.write('\n'+'%'+'model \tmintheta \tmaxtheta \tdelta \teta \tdistmodel')
        self.fQ1.write('\n'+'%'+'%.6g\t\t%.6g\t\t%.6g\t\t%.6g\t\t%.6g\t' % (
            self.wettClass, round(self.minthetai*180/np.pi,3), round(self.maxthetai*180/np.pi,3), self.delta, self.eta ,) + str(self.distModel),)
        self.fQ1.write('\nmintheta \tmaxtheta \tmean  \tstd')
        self.fQ1.write('\n'+'%'+'%3.6g\t\t%3.6g\t\t%3.6g\t\t%3.6g' % (
            round(self.contactAng.min()*180/np.pi,3), round(self.contactAng.max()*180/np.pi,3), round(self.contactAng.mean()*180/np.pi,3), round(self.contactAng.std()*180/np.pi,3)))
        self.fQ1.write('\n======================================================================')
        self.fQ1.write("\n"+"%"+"Sw\t qW(m3/s)\t krw\t qNW(m3/s)\t krnw\t Pc\t Invasions")

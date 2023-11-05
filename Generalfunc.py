import numpy as np
import numpy_indexed as npi
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import pypardiso
from itertools import chain
import warnings
import config
#from joblib import Parallel, delayed
from time import time
import numba
from solver import Solver


def matrixSolver1(Amatrix, Cmatrix):
    return spsolve(Amatrix, Cmatrix)

def matrixSolver(Amatrix, Cmatrix):
    return Solver(Amatrix, Cmatrix).solve()
    #return spsolve(Amatrix, Cmatrix)


def matrixSolverOld(Amatrix, Cmatrix, indP):
    # udut, ipiv, x, info = dsysv(Amatrix, Cmatrix)
    try:
        amax = Amatrix.max()
        Amatrix = Amatrix/amax
        Cmatrix = Cmatrix/amax
        x = pypardiso.spsolve(Amatrix, Cmatrix)
        if any(x > 1.0001):
            #print(x[x > 1.0001])
            ind = config.PPConData[indP[x > 1.0001]]
            arrIndP = np.full(config.nPores+2, False)
            arrIndP[indP] = True
            arrIndP[[-1, 0]] = False

            try:
                m = np.array([npi.indices(indP, i[
                    arrIndP[i]]) for i in ind], dtype=object)
            except KeyError:
                print('Key error')
                from IPython import embed; embed()
            #print(m)
            xmod = np.array([x[i.astype('int')].min() for i in m])
            x[x > 1.0001] = xmod
            #print(xmod)
    except ValueError:
        x = np.zeros([])

    #print(Amatrix.todense()@x - Cmatrix)
    
    #print(indP)
    #print(x)
    #input('waitttt')

    #x = linalg.solve(Amatrix.todense(), Cmatrix)
    #x = pypardiso.spsolve(Amatrix, Cmatrix)
    #x = PETScLinearSolver.solve(Amatrix, Cmatrix)
    #print('Im done with the matrix solver')
    return x


def createFilmsOld(arrr, halfAng, Pc, conAng, conAngRec, conAngAdv, m_exists,
                m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
                recPc, m_initedApexDist):

    cond = (~(m_exists & m_inited).T*arrr).T
    try:
        assert cond.sum() > 0
        conAng[cond] = conAngRec[cond]
        condf = cond & (conAng < (np.pi/2 - halfAng))

        assert condf.sum() > 0
        m_exists[condf] = True
        m_initedApexDist[condf] = np.maximum((config.sigma/Pc[condf]*np.cos(
            conAng[condf]+halfAng[condf])/np.sin(halfAng[condf])), 0.0)
        advPc[condf] = np.where(m_initedApexDist[condf] != 0.0, config.sigma*np.cos(
            np.minimum(np.pi, conAngAdv[condf])+halfAng[condf])/(
            m_initedApexDist[condf]*np.sin(halfAng[condf])), 0.0)
        recPc[condf] = np.where(m_initedApexDist[condf] != 0.0, config.sigma*np.cos(
            np.minimum(np.pi, conAngRec[condf])+halfAng[condf])/(
            m_initedApexDist[condf]*np.sin(halfAng[condf])), 0.0)
        m_inited[condf] = True

        condu = condf & (Pc > m_initOrMaxPcHist)
        assert condu.sum() > 0
        m_initOrMinApexDistHist[condu] = m_initedApexDist[condu]
        m_initOrMaxPcHist[condu] = Pc[condu]
    except AssertionError:
        pass


def createFilms(arr, arrr, halfAng, Pc, conAngRec, conAngAdv, m_exists,
                m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
                recPc, m_initedApexDist):

    cond = (~(m_exists & m_inited).T*arrr).T
    try:
        assert cond.sum() > 0
        conAng = conAngRec[arr] if config.is_oil_inj else conAngAdv[arr]
        condf = cond & (conAng[:, np.newaxis] < (np.pi/2 - halfAng))

        assert condf.sum() > 0
        m_exists[condf] = True
        m_initedApexDist[condf] = np.maximum((config.sigma/Pc[condf]*np.cos(
            (conAng[:, np.newaxis]+halfAng)[condf])/np.sin(halfAng[condf])),
            0.0)

        advPc[condf] = np.where(m_initedApexDist[
            condf] != 0.0, config.sigma*np.cos((np.minimum(np.pi, conAngAdv[
             arr, np.newaxis])+halfAng)[condf])/(
            m_initedApexDist[condf]*np.sin(halfAng[condf])), 0.0)

        recPc[condf] = np.where(m_initedApexDist[
            condf] != 0.0, config.sigma*np.cos((np.minimum(np.pi, conAngRec[
             arr, np.newaxis])+halfAng)[condf])/(
            m_initedApexDist[condf]*np.sin(halfAng[condf])), 0.0)
        m_inited[condf] = True

        condu = condf & (Pc > m_initOrMaxPcHist)
        assert condu.sum() > 0
        m_initOrMinApexDistHist[condu] = m_initedApexDist[condu]
        m_initOrMaxPcHist[condu] = Pc[condu]
    except AssertionError:
        pass


def initCornerApex(arr, arrr, halfAng, conAngRec, conAngAdv, m_exists, m_inited, recPc,
                   advPc, m_initedApexDist, trapped):
    
    #cond = np.array([m_exists[i]*(~trapped[i]) for i in range(trapped.size)])
    cond =  (m_exists.T & (~trapped)).T
    m_inited[cond] = True
    Pc = config.sigma*np.cos(np.minimum(np.pi, conAngRec[
        arr, np.newaxis]+halfAng))/(m_initedApexDist*np.sin(halfAng))
    recPc[cond & (recPc < Pc)] = Pc[cond & (recPc < Pc)]
    advPc[cond] = config.sigma*np.cos((np.minimum(np.pi, conAngAdv[
        arr, np.newaxis])+halfAng)[cond])/(m_initedApexDist[cond]*np.sin(halfAng[cond]))

    #return m_inited, recPc, advPc


def finitCornerApex(arr, arrr, halfAng, Pc, conAngRec, conAngAdv, m_exists,
                    m_inited, m_initOrMaxPcHist, m_initOrMinApexDistHist,
                    advPc, recPc, apexDist, m_initedApexDist, hingAng, trapped):

    cond = (m_inited.T+(~trapped)).T & m_exists
    conAng = conAngRec.copy() if config.is_oil_inj else conAngAdv.copy()
    
    conAng, apexDist = cornerApex(
        arr, arrr, halfAng, Pc, conAng, conAngRec, conAngAdv,
        cond, m_initOrMaxPcHist, m_initOrMinApexDistHist,
        advPc, recPc, apexDist, m_initedApexDist, hingAng)
    
    recPc[cond] = config.sigma*np.cos((np.minimum(np.pi, conAngRec[
        arr, np.newaxis])+halfAng)[cond])/(apexDist[cond]*np.sin(halfAng[cond])
                                           )
    advPc[cond] = config.sigma*np.cos((np.minimum(np.pi, conAngAdv[
        arr, np.newaxis])+halfAng)[cond])/(apexDist[cond]*np.sin(halfAng[cond])
                                           )

    cond1 = cond & (Pc > m_initOrMaxPcHist)
    m_initOrMinApexDistHist[cond1] = apexDist[cond1]
    m_inited[cond] = False
    m_initedApexDist[cond] = apexDist[cond]
    try:
        m_initOrMaxPcHist[cond1] = Pc[cond1]
    except (TypeError, IndexError):
        m_initOrMaxPcHist[cond1] = Pc


def cornerApexS(arrr, halfAng, Pc, conAng, m_exists,
                advPc, apexDist, initedApexDist):
    
    '''apexDist[~m_exists] = config.MOLECULAR_LENGTH
    delta = 0.0 if accurat else config._delta
    
    #from IPython import embed; embed()
    # get the apex dist and contact angle
    # condition 1  
    cond1 = arrr & m_exists & (advPc-config._delta <= Pc) & (
            Pc <= recPc+config._delta)      
    try:
        
        #cond1 = (cond1a.T*arrr).T
        #cond1 = cond1a & arrr
        assert cond1.sum() > 0
        part = np.minimum(0.999999, np.maximum(
            Pc*initedApexDist*np.sin(halfAng)/config.sigma,
            -0.999999))
        hingAng[cond1] = np.minimum(np.maximum(
            np.arccos(part[cond1])-halfAng[cond1], -config._delta),
            np.pi+config._delta)
        #from IPython import embed; embed()
        conAng[cond1] = np.minimum(np.maximum(hingAng[cond1], 0.0), np.pi)
        apexDist[cond1] = initedApexDist[cond1]
    except AssertionError:
        pass'''

    #cond2 = arrr & ~cond1 & m_exists & (Pc < advPc)
    cond2 = arrr & m_exists & (Pc < advPc)
    try:
        #assert (arrr & m_exists).sum() == cond.sum()
        assert cond2.sum() > 0
        apexDist[cond2] = (config.sigma/Pc*np.cos(
            conAng+halfAng)/np.sin(halfAng))[cond2]

        cond2 = cond2 & (apexDist < initedApexDist)
        part = np.minimum(0.999999, np.maximum(
            Pc*initedApexDist*np.sin(halfAng)/config.sigma, -0.999999))
        hingAng = np.minimum(np.maximum(np.arccos(part)-halfAng, 0.0), np.pi)
        conAng[cond2] = hingAng[cond2]
        apexDist[cond2] = initedApexDist[cond2]
    except AssertionError:
        print('assertion not correct!')
        #from IPython import embed; embed()

    return conAng, apexDist


def cornerApex(arr, arrr, halfAng, Pc, conAng, conAngRec, conAngAdv, m_exists,
               m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
               recPc, apexDist, initedApexDist, hingAng, accurat=False,
               overidetrapping=False):
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
   
    apexDist[~m_exists] = config.MOLECULAR_LENGTH
    delta = 0.0 if accurat else config._delta

    try:
        assert not overidetrapping
        apexDist[arrr] = initedApexDist[arrr]
        
    except AssertionError:
        pass
    
    #from IPython import embed; embed()
    try:
        conAng = np.array([conAng[arr]]*halfAng.shape[1]).T
    except IndexError:
        conAng = conAng[arr]
    # get the apex dist and contact angle
    # condition 1
    
    try:
        cond1a = m_exists & (advPc-delta <= Pc) & (
            Pc <= recPc+delta)
        cond1 = (cond1a.T*arrr).T
    except ValueError:
        cond1 = (cond1a*arrr)
    try:
        assert cond1.sum() > 0
        part = np.minimum(0.999999, np.maximum(
            Pc*initedApexDist*np.sin(halfAng)/config.sigma,
            -0.999999))
        hingAng[cond1] = np.minimum(np.maximum(
            np.arccos(part[cond1])-halfAng[cond1], -config._delta),
            np.pi+config._delta)
        #from IPython import embed; embed()
        conAng[cond1] = np.minimum(np.maximum(hingAng[cond1], 0.0), np.pi)
        apexDist[cond1] = initedApexDist[cond1]
    except AssertionError:
        pass

    # condition 2
    try:
        cond2 = m_exists & ~cond1a & (Pc < advPc)
        cond2 = (cond2.T*arrr).T
    except ValueError:
        cond2 = (cond2*arrr)
    try:
        assert cond2.sum() > 0
        #print('  cond2a  ')
        #from IPython import embed; embed()
        #conAng[cond2a] = conAngAdv[cond2a]
        conAng[cond2] = ((conAngAdv[arr]*cond2.T).T)[cond2]
        try:
            apexDist[cond2] = (config.sigma/Pc[cond2])*np.cos(
                conAng[cond2]+halfAng[cond2])/np.sin(halfAng[cond2])
        except (TypeError, IndexError):
            apexDist[cond2] = (config.sigma/Pc)*np.cos(
                conAng[cond2]+halfAng[cond2])/np.sin(halfAng[cond2])

        cond2b = (apexDist < initedApexDist) & cond2
        assert cond2b.sum() > 0
        #from IPython import embed; embed()
        #print('  cond2b  ')
        part = np.minimum(0.999999, np.maximum(
            Pc*initedApexDist*np.sin(halfAng)/config.sigma, -0.999999))
        hingAng[cond2b] = np.minimum(np.maximum(
            np.arccos(part[cond2b])-halfAng[cond2b], 0.0), np.pi)
        conAng[cond2b] = hingAng[cond2b]
        apexDist[cond2b] = initedApexDist[cond2b]
    except AssertionError:
        pass

    # condition 3
    cond3a = m_exists & ~cond1 & ~cond2 & (Pc > m_initOrMaxPcHist)
    try:
        cond3 = (cond3a.T*arrr).T
    except ValueError:
        cond3 = (cond3a*arrr)
    try:
        assert cond3.sum() > 0
        # print('cond3: ')
        conAng[cond3] = (np.minimum(np.pi, conAngRec[
            arr, np.newaxis])*cond3)[cond3]
        apexDist[cond3] = config.sigma/Pc*np.cos(
            conAng[cond3]+halfAng[cond3])/np.sin(halfAng[cond3])
    except (ValueError, TypeError):
        print('im in coniton 3')
        from IPython import embed; embed()
        conAng[cond3] = np.minimum(np.pi, conAngRec[cond3])
        apexDist[cond3] = config.sigma/Pc[cond3]*np.cos(
            conAng[cond3]+halfAng[cond3])/np.sin(halfAng[cond3])
    except AssertionError:
        pass

    # condition 4
    cond4a = m_exists & ~cond1 & ~cond2 & ~cond3a & (Pc > recPc)
    try:
        cond4 = (cond4a.T*arrr).T
    except ValueError:
        cond4 = (cond4a*arrr)
    try:
        assert cond4.sum() > 0
        print('  4  ')
        conAng[cond4] = conAngRec[cond4]
        print('Im in condition 4')
        from IPython import embed; embed()
    except AssertionError:
        pass

    cond5 = m_exists & ~cond1 & ~cond2 & ~cond3a & ~cond4a
    try:
        cond5 = (cond5.T*arrr).T
    except ValueError:
        cond5 = (cond5*arrr)
    try:
        assert cond5.sum() > 0
        print('cond5: ', cond5)
        apexDist[cond5] = (config.sigma/Pc)*np.cos(
            conAng[cond5]+halfAng[cond5])/np.sin(halfAng[cond5])
    except AssertionError:
        pass
    
    return conAng, apexDist


def calcAreaWOld(arrr, halfAng, Pc, conAngRec, conAngAdv, m_exists,
              m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
              recPc, initedApexDist, hingAng):

    apexDist = np.zeros(halfAng.shape)
    conAng = conAngRec.copy() if config.is_oil_inj else conAngAdv.copy()
    cornerApex(arrr, halfAng, Pc, conAng, conAngRec, conAngAdv, m_exists,
               m_initOrMaxPcHist, m_initOrMinApexDistHist, advPc,
               recPc, apexDist, initedApexDist, hingAng)

    # -- obtain corner conductance -- #
    dimlessCornerA = np.zeros(halfAng.shape)
    cornerGstar = np.zeros(halfAng.shape)

    cond1 = m_exists & (np.abs(conAng+halfAng-np.pi/2) < 0.01)
    dimlessCornerA[cond1] = np.sin(halfAng[cond1])*np.cos(halfAng[cond1])

    cond2 = m_exists & (np.abs(conAng+halfAng-np.pi/2) >= 0.01)
    dimlessCornerA[cond2] = pow(np.sin(halfAng[cond2])/np.cos(
        conAng[cond2] + halfAng[cond2]), 2.0)*(np.cos(conAng[cond2])*np.cos(
         conAng[cond2] + halfAng[cond2])/np.sin(halfAng[cond2]) + conAng[
          cond2] + halfAng[cond2] - np.pi/2)

    cornerGstar[m_exists] = np.sin(halfAng[m_exists])*np.cos(halfAng[
        m_exists])/(4*pow(1+np.sin(halfAng[m_exists]), 2))
    cornerG = cornerGstar.copy()

    cond3 = m_exists & (np.abs(conAng+halfAng-np.pi/2) > 0.01)
    cornerG[cond3] = dimlessCornerA[cond3]/(4.0*pow(1 - np.sin(
        halfAng[cond3])/np.cos(conAng[cond3] + halfAng[cond3])*(
            conAng[cond3] + halfAng[cond3] - np.pi/2), 2.0))

    cFactor = np.where(cornerG != 0.0, 0.364+0.28*cornerGstar/cornerG, 0.0)
    conductance = cFactor*pow(apexDist, 4)*pow(
        dimlessCornerA, 2)*cornerG/config.muw
    area = apexDist*apexDist*dimlessCornerA

    cornerCond = conductance.sum(axis=1)
    cornerArea = area.sum(axis=1)

    return cornerArea[arrr], cornerCond[arrr]


def calcAreaW(arrr, halfAng, conAng, m_exists, apexDist):
    # -- obtain corner conductance -- #
    dimlessCornerA = np.zeros(halfAng.shape)
    cornerGstar = np.zeros(halfAng.shape)

    cond1 = m_exists & (np.abs(conAng+halfAng-np.pi/2) < 0.01)
    dimlessCornerA[cond1] = np.sin(halfAng[cond1])*np.cos(halfAng[cond1])

    cond2 = m_exists & (np.abs(conAng+halfAng-np.pi/2) >= 0.01)
    dimlessCornerA[cond2] = pow(np.sin(halfAng[cond2])/np.cos(
        conAng[cond2] + halfAng[cond2]), 2.0)*(np.cos(conAng[cond2])*np.cos(
         conAng[cond2] + halfAng[cond2])/np.sin(halfAng[cond2]) + conAng[
          cond2] + halfAng[cond2] - np.pi/2)

    cornerGstar[m_exists] = np.sin(halfAng[m_exists])*np.cos(halfAng[
        m_exists])/(4*pow(1+np.sin(halfAng[m_exists]), 2))
    cornerG = cornerGstar.copy()

    cond3 = m_exists & (np.abs(conAng+halfAng-np.pi/2) > 0.01)
    cornerG[cond3] = dimlessCornerA[cond3]/(4.0*pow(1 - np.sin(
        halfAng[cond3])/np.cos(conAng[cond3] + halfAng[cond3])*(
            conAng[cond3] + halfAng[cond3] - np.pi/2), 2.0))

    cFactor = np.where(cornerG != 0.0, 0.364+0.28*cornerGstar/cornerG, 0.0)
    conductance = cFactor*pow(apexDist, 4)*pow(
        dimlessCornerA, 2)*cornerG/config.muw
    area = apexDist*apexDist*dimlessCornerA

    cornerCond = conductance.sum(axis=1)
    cornerArea = area.sum(axis=1)

    

    return cornerArea[arrr], cornerCond[arrr]


def getIndex(arrr1, arrr2):
    return np.array(npi.indices(arrr1, arrr2), dtype=int)


def getValue(arrIndP, arrIndT, gL):
    c = arrIndP.sum()
    indP = config.poreList[arrIndP[1:-1]]
    Cmatrix = np.zeros(c)
    row, col, data = [], [], []

    # for entries in the diagonals
    def worker(arr):
        return gL[arr[arrIndT[arr]] - 1].sum()
    
    cond = [*map(worker, config.PTConData[arrIndP[:-1]])]   
    m = np.arange(c)
    row.extend(m)
    col.extend(m)
    data.extend(cond)

    arrT = arrIndT[1:] & arrIndP[config.P1array] & arrIndP[config.P2array]
    cond = -gL[arrT]
    j = npi.indices(indP, config.P1array[arrT])
    k = npi.indices(indP, config.P2array[arrT])
    row.extend(j)
    col.extend(k)
    data.extend(cond)
    row.extend(k)
    col.extend(j)
    data.extend(cond)

    # for entries on/in the inlet boundary
    arrT = arrIndT[1:] & config.T_isOnInletBdr
    arrP = config.P1array[arrT]*(arrIndP[config.P1array[arrT]]) +\
        config.P2array[arrT]*(arrIndP[config.P2array[arrT]])
    cond = gL[arrT]
    m = npi.indices(indP, arrP)

    Cmatrix = np.array([*map(lambda i: cond[m == i].sum(), range(c))])
    #Cmatrix = np.array([cond[m == i].sum() for i in range(c)])
    #from IPython import embed; embed()
    Amatrix = csc_matrix((data, (row, col)), shape=(c, c),
                         dtype=float)
    #from IPython import embed; embed()
    return Amatrix, Cmatrix


def isConnected(indPS, indTS):
    connectedP = np.zeros(config.nPores+2, dtype='bool')
    connectedT = np.zeros(config.nThroats+1, dtype='bool')

    doneP = np.ones(config.nPores+2, dtype='bool')
    doneP[indPS] = False
    doneP[0] = False
    doneT = np.ones(config.nThroats+1, dtype='bool')
    doneT[indTS] = False
    tin = list(config.conTToIn[~doneT[config.conTToIn]])

    while True:
        try:
            connP = np.zeros(config.nPores+2, dtype='bool')
            connT = np.zeros(config.nThroats+1, dtype='bool')
            doneP[0] = False
            t = tin.pop(0)
            while True:
                doneT[t] = True
                connT[t] = True
                p = np.array([config.P1array[t-1], config.P2array[t-1]])

                p = p[~doneP[p]]
                doneP[p] = True
                connP[p] = True

                try:
                    tt = np.zeros(config.nThroats+1, dtype='bool')
                    tt[np.array([*chain(*config.PTConData[p])])] = True
                    t = config.throatList[tt[1:] & ~doneT[1:]]
                    assert t.size > 0
                except (AssertionError, IndexError):
                    try:
                        tin = np.array(tin)
                        tin = list(tin[~doneT[tin]])
                    except IndexError:
                        tin=[]
                    break
            try:
                assert connP[0]
                connectedP[connP] = True
                connectedT[connT] = True
            except AssertionError:
                pass
        except (AssertionError, IndexError):
            break

    connectedP = connectedP & config.P_isinsideBox
    connectedT[1:] = connectedT[1:] & config.T_isinsideBox
    return connectedP, connectedT

def isConnected1(indPS, indTS):
    connectedP = np.zeros(config.nPores+2, dtype='bool')
    connectedT = np.zeros(config.nThroats+1, dtype='bool')

    doneP = np.ones(config.nPores+2, dtype='bool')
    doneP[indPS] = False
    doneT = np.ones(config.nThroats+1, dtype='bool')
    doneT[indTS] = False

    tin = list(config.conTToIn[~doneT[config.conTToIn]])

    while True:
        try:
            assert len(tin) > 0
            t = tin.pop(0)
            while True:
                doneT[t] = True
                connectedT[t] = True
                p = np.array([config.P1array[t-1], config.P2array[t-1]])

                p = p[~doneP[p]]
                doneP[p] = True
                connectedP[p] = True

                try:
                    assert p.size > 0
                    tt = np.zeros(config.nThroats+1, dtype='bool')
                    tt[np.array([*chain(*config.PTConData[p])])] = True
                    t = config.throatList[tt[1:] & ~doneT[1:]]
                    assert t.size > 0
                except AssertionError:
                    try:
                        assert len(tin) > 0
                        tin = np.array(tin)
                        tin = list(tin[~doneT[tin]])
                    except AssertionError:
                        break
                    break
        except AssertionError:
            break

    connectedP = connectedP & config.P_isinsideBox
    connectedT[1:] = connectedT[1:] & config.T_isinsideBox
    return connectedP, connectedT


def isTrappedP1(i, fluid):
    try:
        assert config.trappedP[i - 1]
        return True
    except AssertionError:
        cond1 = ((config.fluidP == 0) | (config.GarrayP <= config.bndG2)) * (
            fluid == 0) + (config.fluidP == 1) * (fluid == 1)
        indPS = cond1 & ~(config.trappedP)
        cond2 = ((config.fluidT == 0) | (config.GarrayT <= config.bndG2)) * (
            fluid == 0) + (config.fluidT == 1) * (fluid == 1)
        indTS = cond2 & ~(config.trappedT)

        arrT = np.full(config.nThroats, False)
        arrP = np.full(config.nPores, False)
        tttt = arrT.copy()
        tttt[config.PTConData[i] - 1] = True
        thr = tttt & indTS
        indP = arrP.copy()
        indP[i-1] = True
        indT = thr.copy()
        indexP = indP.copy()
        indexT = thr.copy()
        pppp = indP.copy()

        while any(indP) or any(indT):
            indP = np.full(config.nPores + 2, False)
            indP[config.P1array[indT]] = True
            indP[config.P2array[indT]] = True
            indP = indP[1:-1] & ~pppp
            pppp[indP] = True
            indP = indP & indPS
            indexP[indP] = True

            indT = arrT.copy()
            indP = config.poreList[indP]
            try:
                indT[np.array([*chain(*config.PTConData[indP])]) - 1] = True
                indT = indT & ~tttt
                tttt[indT] = True
                indT = indT & indTS
                indexT[indT] = True
            except IndexError:
                pass

            thr1 = config.arrConTToOut[1:] & indexT
            thr2 = config.arrConTToIn[1:] & indexT
            try:
                assert any(thr1 | thr2)
                return False
            except AssertionError:
                pass
            
        config.trappedP[indexP] = True
        config.trappedT[indexT] = True
        cnt = max(config.clusterP.max(), config.clusterT.max()) + 1
        config.clusterP[indexP] = cnt
        config.clusterT[indexT] = cnt

    return True

def isTrappedP(i, fluid):
    try:
        assert config.trappedP[i - 1]
        return True
    except AssertionError:
        indPS = np.ones(config.nPores+2, dtype='bool')
        try:
            assert fluid == 0
            indPS[1:-1] =  (~config.trappedP) & ((config.fluidP==0) | (
                config.GarrayP <= config.bndG2))
            indTS =  (~config.trappedT) & ((config.fluidT==0) | (
                config.GarrayT <= config.bndG2))
        except AssertionError:
            indPS[1:-1] =  (~config.trappedP) & (config.fluidP==1)
            indTS =  (~config.trappedT) & (config.fluidT==1)

        indPS[i] = False
        
        ttlist = list(config.PTConData[i][indTS[config.PTConData[i]-1]]+config.nPores)
        indTS[config.PTConData[i]-1] = False
        pplist = []
        indexP, indexT = [i], []
        while True:
            try:
                t = ttlist.pop(np.argmin(config.distToBoundary[ttlist]))
                indexT.append(t)
                pp = np.array([config.P1array[t-config.nPores-1], config.P2array[
                    t-config.nPores-1]])
                pplist.extend(pp[indPS[pp]])
                indPS[pp] = False
            except ValueError:
                pass
            try:
                p = pplist.pop(np.argmin(config.distToBoundary[pplist]))
                try:
                    assert p <= 0
                    return False
                except AssertionError:
                    indexP.append(p)
                ttlist.extend(config.PTConData[p][indTS[config.PTConData[p]-1]]+config.nPores)
                indTS[config.PTConData[p]-1] = False
            except ValueError:
                pass
            try:
                assert len(pplist)+len(ttlist) > 0
            except AssertionError:
                try:
                    config.trappedP[np.array(indexP)-1] = True
                except IndexError:
                    pass
                try:
                    config.trappedT[np.array(indexT)-config.nPores-1] = True
                except IndexError:
                    pass
                
                return True
                

def isTrappedT(i, fluid):
    try:
        assert config.trappedT[i - 1]
        return True
    except AssertionError:
        try:
            assert config.P1array[i - 1] * config.P2array[i - 1] <= 0
            return False
        except AssertionError:
            indPS = np.ones(config.nPores+2, dtype='bool')
            try:
                assert fluid == 0
                indPS[1:-1] =  (~config.trappedP) & (((config.fluidP==0) | (
                    config.GarrayP <= config.bndG2)))
                indTS =  (~config.trappedT) & ((config.fluidT==0) | (
                    config.GarrayT <= config.bndG2))
            except AssertionError:
                indPS[1:-1] =  (~config.trappedP) & (config.fluidP==1)
                indTS =  (~config.trappedT) & (config.fluidT==1)

            indTS[i-1] = False
            
            pp = np.array([config.P1array[i-1], config.P2array[i-1]])
            pplist = list(pp[indPS[pp]])
            indPS[pp] = False
            ttlist = []
            indexP, indexT = [], [i]

            while True:
                try:
                    p = pplist.pop(np.argmin(config.distToBoundary[pplist]))
                    indexP.append(p)
                    try:
                        assert p <= 0
                        return False
                    except AssertionError:
                        pass
                    
                    ttlist.extend(config.PTConData[p][indTS[config.PTConData[p]-1]]+config.nPores)
                    indTS[config.PTConData[p]-1] = False
                except ValueError:
                    pass
                try:
                    
                    t = ttlist.pop(np.argmin(config.distToExit[ttlist]))
                    indexT.append(t)
                    pp = np.array([config.P1array[t-config.nPores-1], config.P2array[
                        t-config.nPores-1]])
                    pplist.extend(pp[indPS[pp]])
                    indPS[pp] = False
                except ValueError:
                    pass
                try:
                    assert len(pplist)+len(ttlist) > 0
                except AssertionError:
                    try:
                        config.trappedP[np.array(indexP)-1] = True
                    except IndexError:
                        pass
                    try:
                        config.trappedT[np.array(indexT)-config.nPores-1] = True
                    except IndexError:
                        pass
                    return True             


def isTrappedT1(i, fluid):
    try:
        assert config.trappedT[i - 1]
        return True
    except AssertionError:
        try:
            assert config.P1array[i - 1] * config.P2array[i - 1] <= 0
            return False
        except AssertionError:
            indPS =  (~config.trappedP) & (((config.fluidP==0) | (
                config.GarrayP <= config.bndG2)) if fluid == 0 else (config.fluidP==1))
            indTS =  (~config.trappedT) & (((config.fluidT==0) | (
                config.GarrayT <= config.bndG2)) if fluid == 0 else (config.fluidT==1))

            arrT = np.zeros(config.nThroats, dtype='bool')
            arrP = np.zeros(config.nPores, dtype='bool')
            p = np.zeros(config.nPores + 2, dtype='bool')

            p[[config.P1array[i-1], config.P2array[i-1]]] = True
            p = p[1:-1] & indPS
            try:
                assert not any(p)
                config.trappedT[i - 1] = True
                return True
            except AssertionError:
                pass
            p = config.poreList[p]
            indP = arrP.copy()
            indP[p - 1] = True
            indT = arrT.copy()
            indexP = indP.copy()
            indexT = arrT.copy()
            indexT[i - 1] = True
            tttt = indexT.copy()
            pppp = np.full(config.nPores + 2, False)
            pppp[config.pin_] = True
            pppp[p] = True

            while any(indP) or any(indT):
                indT = arrT.copy()
                indP = config.poreList[indP]
                try:
                    indT[np.array([*chain(*config.PTConData[indP])]) - 1] = True
                    indT = indT & ~tttt
                    tttt[indT] = True
                    indT = indT & indTS
                    indexT[indT] = True

                    indP = np.full(config.nPores + 2, False)
                    indP[config.P1array[indT]] = True
                    indP[config.P2array[indT]] = True
                    indP = indP & ~pppp
                    pppp[indP] = True
                    indP = indP[1:-1] & indPS
                    indexP[indP] = True
                except IndexError:
                    pass

                thr1 = config.arrConTToOut[1:] & indexT
                thr2 = config.arrConTToIn[1:] & indexT
                try:
                    assert any(thr1 | thr2)
                    return False
                except AssertionError:
                    pass
        
            config.trappedP[indexP] = True
            config.trappedT[indexT] = True
            cnt = max(config.clusterP.max(), config.clusterT.max()) + 1
            config.clusterP[indexP] = cnt
            config.clusterT[indexT] = cnt

    return True


def Saturation(AreaWP, AreaWT, AreaSP_P, AreaSP_T):
    #if (AreaWP < 0).sum() + (AreaWT < 0).sum() > 0:
    #    from IPython import embed; embed()
    satW_P = AreaWP/AreaSP_P
    satW_T = AreaWT/AreaSP_T
    num = (satW_P[config.P_isinsideBox[1:-1]]*config.volarrayP[
            config.P_isinsideBox[1:-1]]).sum() + (satW_T[
             config.T_isinsideBox]*config.volarrayT[config.T_isinsideBox]).sum()

    return num/config.totVoidVolume


def computegL(gP, gT):
    gL = np.zeros(config.nThroats)
    cond = (gT > 0.0) & ((gP[config.P1array] > 0) | (config.P1array < 1)) & (
        (gP[config.P2array] > 0) | (config.P2array < 1))
    cond3 = cond & (gP[config.P1array] > 0) & (gP[config.P2array] > 0)
    cond2 = cond & (gP[config.P1array] == 0) & (gP[config.P2array] > 0) & (
        config.LP2array_mod > 0)
    cond1 = cond & (gP[config.P1array] > 0) & (gP[config.P2array] == 0) & (
        config.LP1array_mod > 0)

    gL[cond3] = 1/(config.LP1array_mod[cond3]/gP[config.P1array[cond3]] +
                   config.LTarray_mod[cond3]/gT[cond3] + config.LP2array_mod[
                   cond3]/gP[config.P2array[cond3]])
    gL[cond2] = 1/(config.LTarray_mod[cond2]/gT[cond2] + config.LP2array_mod[
                   cond2]/gP[config.P2array[cond2]])
    gL[cond1] = 1/(config.LTarray_mod[cond1]/gT[cond1] + config.LP1array_mod[
                   cond1]/gP[config.P1array[cond1]])

    return gL


def computeFlowrate(gL, phase=0):
    arrPoreList = np.zeros(config.nPores+2, dtype='bool')
    arrPoreList[config.P1array[(gL > 0.0)]] = True
    arrPoreList[config.P2array[(gL > 0.0)]] = True
    indPS = config.poreList[arrPoreList[1:-1]]
    indTS = config.throatList[(gL > 0.0)]
    indP, indT = isConnected(indPS, indTS)
    Amatrix, Cmatrix = getValue(indP, indT, gL)
    
    '''try:
        assert phase == 0
        config.presW[config.P_isOnInletBdr] = 1.0
        config.presW[indP] = matrixSolver(Amatrix, Cmatrix)
        delP = np.abs(config.presW[config.P1array] - config.presW[config.P2array])
    except AssertionError:
        config.presNW[config.P_isOnInletBdr] = 1.0
        config.presNW[indP] = matrixSolver(Amatrix, Cmatrix)
        delP = np.abs(config.presNW[config.P1array] - config.presNW[config.P2array])'''

    pres = np.zeros(config.nPores+2)
    pres[config.P_isOnInletBdr] = 1.0
    pres[indP] = matrixSolver(Amatrix, Cmatrix)
    delP = np.abs(pres[config.P1array] - pres[config.P2array])
    qp = gL*delP
    qinto = qp[config.T_isOnInletBdr & indT[1:]].sum()
    qout = qp[config.T_isOnOutletBdr & indT[1:]].sum()
    try:
        assert np.isclose(qinto, qout, atol=1e-30)
        qout = (qinto+qout)/2
    except AssertionError:
        pass

    return qout



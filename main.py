from datetime import date
import sys
from inputData import InputData
from network import Network
from SPhase import SinglePhase
from TPhaseD import TwoPhaseDrainage
#from TPhaseImb import TwoPhaseImbibition


# __DATE__ = "Jul 25 , 2023"
__DATE__ = date.today().strftime("%b") + " " + str(date.today().day) + ", " +\
      str(date.today().year)


def main():
    try:
        input_file_name = ""

        print("\nNetwork Model Code version 2 alpha, built: ", __DATE__, "\n")

        if len(sys.argv) > 1:
            input_file_name = sys.argv[1]
        else:
            input_file_name = input("Please input data file : ")

        input_data = InputData(input_file_name)
        netsim = Network(input_file_name)

        # Single Phase computation
        netsim = SinglePhase(netsim)
        netsim.singlephase()

        # two Phase simulations
        if input_data.satControl():
            for cycle in range(len(input_data.satControl())):
                netsim.finalSat, Pc, netsim.dSw, netsim.minDeltaPc,\
                 netsim.deltaPcFraction, netsim.calcKr, netsim.calcI,\
                 netsim.InjectFromLeft, netsim.InjectFromRight,\
                 netsim.EscapeFromLeft, netsim.EscapeFromRight =\
                 input_data.satControl()[cycle]
                netsim.filling = True
                if netsim.finalSat < netsim.satW:
                    # Drainage process
                    (netsim.wettClass, netsim.minthetai, netsim.maxthetai,
                     netsim.delta, netsim.eta, netsim.distModel, netsim.sepAng
                     ) = input_data.initConAng('INIT_CONT_ANG')
                    netsim.is_oil_inj = True
                    netsim.maxPc = Pc
                    netsim = TwoPhaseDrainage(netsim)
                    #from IPython import embed; embed()
                    netsim.drainage()
                else:
                    # Imbibition process
                    netsim.probable = False
                    (netsim.wettClass, netsim.minthetai, netsim.maxthetai,
                     netsim.delta, netsim.eta, netsim.distModel, netsim.sepAng
                     ) = input_data.initConAng('EQUIL_CON_ANG')
                    netsim.minPc = Pc
                    netsim = TwoPhaseImbibition(netsim)
                    netsim.imbibition()
        else:
            pass
    except Exception as exc:
        print("\n\n Exception on processing: \n", exc, "Aborting!\n")
        return 1
    except:
        from IPython import embed; embed()
        print("\n\n Unknown exception! Aborting!\n")
        return 1

    return 0





if __name__ == "__main__":
    sys.exit(main())



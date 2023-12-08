import numpy as np

class ACPricing:
    def __init__(self, F0=100, F=100, BI=85, BO=103, r=0.03, TotalStep=251, 
                 PastedStep=1, coupon=0.2, q=0.2, NRepeat=500000, u=0, sigma=0.13, t=1,
                 IsKnockInNow=0, IsKnockOutNow=0, BOObsrvStartDate=21, BOObsrvFreq=21, dS=0.0005, dV=0.0005,dR = 0.0005):
        # Parameters
        self.F0 = F0
        self.F = F
        self.BI = BI
        self.BO = BO
        self.r = r
        self.TotalStep = TotalStep
        self.PastedStep = PastedStep
        self.coupon = coupon
        self.q = q
        self.NRepeat = NRepeat
        self.u = u
        self.sigma = sigma
        self.t = t
        self.IsKnockInNow = IsKnockInNow
        self.IsKnockOutNow = IsKnockOutNow
        self.BOObsrvStartDate = BOObsrvStartDate
        self.BOObsrvFreq = BOObsrvFreq
        self.BOObsrvDate = np.arange(self.BOObsrvStartDate, TotalStep + 2, self.BOObsrvFreq) - PastedStep
        self.NStep = TotalStep - PastedStep
        self.h = t/self.NStep
        self.dS = dS
        self.dV = dV
        self.dR = dR
        self.generate_paths()
    
    def generate_paths(self):
        self.X = np.random.randn(self.NRepeat, self.NStep)
        dlogF = (self.u - self.sigma**2/2)*self.h + self.sigma*np.sqrt(self.h)*self.X
        self.paths = self.F * np.hstack([np.ones((self.NRepeat,1)), np.exp(np.cumsum(dlogF, axis=1))])
    
    def check_knock_out(self, paths, PastedStep=1):
        IsKnockOut = paths >= self.BO
        # set all values before the pasted step to be 0
        IsKnockOut[:, :PastedStep-1] = 0
        # set all values not in the observation dates to be 0
        mask = np.zeros(IsKnockOut.shape[1], dtype=bool)
        BOObsrvDate = np.arange(self.BOObsrvStartDate, self.TotalStep + 2, self.BOObsrvFreq) - PastedStep
        mask[BOObsrvDate-1] = True
        IsKnockOut[:, ~mask] = 0
        return IsKnockOut
    
    def get_discounted_payoff(self, paths, PastedStep=1):
        IsKnockOut = self.check_knock_out(paths, PastedStep)
        IsKnockIn = paths <= self.BI
        DiscPayoff = np.zeros(self.NRepeat)

        case1Num = 0
        case2Num = 0
        case3Num = 0
        case4Num = 0

        for i in range(self.NRepeat):
            if np.sum(IsKnockOut[i, :]):
                KnockOutT = np.where(IsKnockOut[i, :])[0][0] + PastedStep
                DiscPayoff[i] = np.exp(-self.r * KnockOutT / 252) * self.F0 * (1 + self.coupon * KnockOutT / 252)
                case1Num += 1
            elif (not np.sum(IsKnockIn[i, :])) and (self.IsKnockInNow == 0):
                DiscPayoff[i] = np.exp(-self.r * self.NStep / 252) * self.F0 * (1 + self.coupon)
                case2Num += 1
            elif paths[i, -1] > self.F0:
                DiscPayoff[i] = np.exp(-self.r * self.NStep / 252) * self.F0
                case3Num += 1
            else:
                DiscPayoff[i] = np.exp(-self.r * self.NStep / 252) * paths[i, -1]
                case4Num += 1

        self.case1Num = case1Num
        self.case2Num = case2Num
        self.case3Num = case3Num
        self.case4Num = case4Num
        
        return DiscPayoff
    
    # Now, let's integrate the helper methods into the main methods:
    def compute_PV(self, PastedStep=1):
        if self.IsKnockOutNow:
            PV = np.exp(-self.r * self.PastedStep / 252) * self.F0 * (1 + self.coupon * self.PastedStep / 252)
        else:
            DiscPayoff = self.get_discounted_payoff(self.paths, PastedStep)
            PV = np.mean(DiscPayoff)
            
            # Ensure all paths are considered
            if len(DiscPayoff) != self.NRepeat:
                raise ValueError("There is a discrepancy in path numbers!")
            
        self.PV = PV

    # Continue refactoring compute_theta
    def compute_theta(self):
        TPastedStep = self.PastedStep + 1
        
        if self.IsKnockOutNow:
            return 0
        else:
            # drop the second column
            paths = np.hstack([self.paths[:, 0].reshape(-1, 1), self.paths[:, 2:]])
            print(paths.shape)
            TDiscPayoff = self.get_discounted_payoff(paths, TPastedStep)
            print(TDiscPayoff)
            TPV = np.mean(TDiscPayoff)
            return TPV - self.PV

   
    # Continue refactoring compute_delta_gamma
    def compute_delta_gamma(self):
        paths = self.paths
        FUp = self.F * (1 + self.dS)
        FDown = self.F * (1 - self.dS)

        paths_up = paths * FUp/self.F
        paths_down = paths * FDown/self.F

        PVUp = np.mean(self.get_discounted_payoff(paths_up))
        PVDown = np.mean(self.get_discounted_payoff(paths_down))
        
        Delta = (PVUp - PVDown) / (self.F * self.dS * 2)
        Gamma = (PVUp + PVDown - 2 * self.PV) / (self.F * self.dS)**2
        
        return Delta, Gamma

    # Continue refactoring compute_vega
    def compute_vega(self):
        paths = self.paths
        
        sigmaUp = self.sigma * (1 + self.dV)
        sigmaDown = self.sigma * (1 - self.dV)
        
        dlogFUp = (self.u - sigmaUp**2/2)*self.h + sigmaUp*np.sqrt(self.h)*self.X
        dlogFDown = (self.u - sigmaDown**2/2)*self.h + sigmaDown*np.sqrt(self.h)*self.X
        
        paths_up = self.F * np.hstack([np.ones((self.NRepeat,1)), np.exp(np.cumsum(dlogFUp, axis=1))])
        paths_down = self.F * np.hstack([np.ones((self.NRepeat,1)), np.exp(np.cumsum(dlogFDown, axis=1))])
        
        VPVUp = np.mean(self.get_discounted_payoff(paths_up))
        VPVDown = np.mean(self.get_discounted_payoff(paths_down))
        
        Vega = (VPVUp - VPVDown) / (self.sigma * self.dV * 2) / 100
        return Vega

    def compute_greeks(self):
        self.Delta, self.Gamma = self.compute_delta_gamma()
        self.Theta = self.compute_theta()
        self.Vega = self.compute_vega()

    def evaluate(self):
        self.generate_paths()
        self.compute_PV()
        self.compute_greeks()

        self.case1Ratio = self.case1Num / self.NRepeat
        self.case2Ratio = self.case2Num / self.NRepeat
        self.case3Ratio = self.case3Num / self.NRepeat
        self.case4Ratio = self.case4Num / self.NRepeat
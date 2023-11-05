def isconnected(self):
        iElements = set([-1])
        for ind in self.elem[-1].neighbours:
            try:
                assert ind not in iElements and not self.elem[ind].connected
                frontier, connected = set(self.elem[ind].neighbours), set([-1, ind])
                iElements.add(ind)

                while frontier:
                    old_frontier = frontier
                    frontier = set()

                    for j in old_frontier:
                        try:
                            assert j not in iElements
                            connected.add(j)
                            iElements.add(j)
                            
                            try:
                                assert j != 0
                                for i in self.elem[j].neighbours:
                                    frontier.add(i)
                            except AssertionError:
                                pass
                        except AssertionError:
                            pass
                            
                try:
                    assert 0 in connected
                    for j in connected:
                        try:
                            assert not self.elem[j].connected
                            setattr(self.elem[j], "connected", True)
                        except AssertionError:
                            pass
                except AssertionError:
                    pass
            except AssertionError:
                pass


def computeHalfAng(self, elem):
        angle = -12*sqrt(3)*elem.G
        assert (abs(angle) <= 1.0)

        beta2_min = atan(2/sqrt(3)*cos((acos(angle)/3)+(
            4*pi/3)))
        beta2_max = atan(2/sqrt(3)*cos(acos(angle)/3))

        randNum = random.randint(0, config.RAND_MAX)/config.RAND_MAX
        randNum = 0.5*(randNum+0.5)

        beta2 = beta2_min + (beta2_max - beta2_min)*randNum
        beta1 = 0.5*(asin((tan(beta2) + 4*elem.G) * sin(
            beta2) / (tan(beta2) - 4*elem.G)) - beta2)
        beta3 = pi/2 - beta2 - beta1

        assert (beta1 <= beta2) & (beta2 <= beta3)
        elem.halfAng = np.array([beta1, beta2, beta3])

def isOnBdr(self, el):
        #for el in self.elemThroat:
        try:
            assert not el.isPore and el.isinsideBox
            try:
                assert ((not self.elem[el.P1].isinsideBox) & (self.elem[
                    el.P1].x <= self.xstart)) | ((not self.elem[el.P2].isinsideBox) & (
                        self.elem[el.P2].x <= self.xstart))
                setattr(el, 'isOnInletBdr', True)
            except AssertionError:
                pass
            try:
                assert ((not self.elem[el.P1].isinsideBox) & (self.elem[
                el.P1].x >= self.xend)) | ((not self.elem[el.P2].isinsideBox) & (
                    self.elem[el.P2].x >= self.xend))
                setattr(el, "isOnOutletBdr", True)
            except AssertionError:
                pass
            
            try:
                assert not self.elem[el.P2].isinsideBox
                try:
                    assert ((self.elem[
                    el.P2].x < self.xstart) | (el.P2 == self.pin_))
                    setattr(self.elem[el.P2], "isOnInletBdr", True)
                except AssertionError:
                    pass
                try:
                    assert ((self.elem[
                    el.P2].x > self.xend) | (el.P2 == self.pout_))
                    setattr(self.elem[el.P2], "isOnOutletBdr", True)
                except AssertionError:
                    pass
            except AssertionError:
                pass
            
            try:
                assert not self.elem[el.P1].isinsideBox
                try:
                    assert ((self.elem[
                    el.P1].x < self.xstart) | (el.P1 == self.pin_))
                    setattr(self.elem[el.P1], "isOnInletBdr", True)
                except AssertionError:
                    pass 
                try:
                    assert ((self.elem[
                    el.P1].x > self.xend) | (el.P1 == self.pout_))          
                    setattr(self.elem[el.P1], "isOnOutletBdr", True)
                except AssertionError:
                    pass
            except AssertionError:
                pass
            
        except AssertionError:
            pass

def modifyLength1(self):
        for el in self.elemThroat:
            try:
                assert el.isinsideBox and not self.elem[el.P2].isinsideBox
                try:
                    try:
                        assert el.P2 > 0
                        scaleFact = (self.elem[el.P2].x-self.elem[el.P1].x)/(el.LT+el.LP1+el.LP2)
                    except AssertionError:
                        scaleFact = (self.elem[el.P2].x-self.elem[el.P1].x)/abs(
                            self.elem[el.P2].x-self.elem[el.P1].x)

                    throatStart = self.elem[el.P1].x + el.LP1*scaleFact
                    throatEnd = throatStart + el.LT*scaleFact
                    bdr = self.xstart if (self.elem[el.P2].x < self.xstart) else self.xend           

                    #from IPython import embed; embed()
                    cond1a = (el.P2 < 1)
                    cond1b = (not cond1a) & (throatEnd > self.xstart) & (throatEnd < self.xend)
                    cond1c = (not cond1a) & (not cond1b) & (throatStart > self.xstart) & (
                        throatStart < self.xend)
                    cond1d = not (cond1a | cond1b | cond1c)

                    try:
                        assert (cond1a | cond1c | cond1d)
                        setattr(el, "LP2mod", 0.0)
                    except AssertionError:
                        if cond1b: setattr(el, "LP2mod", (bdr-throatEnd)/scaleFact)
                
                    if cond1c: setattr(el, "LTmod", (bdr-throatStart)/scaleFact)
                    if cond1d: setattr(el, "LTmod", 0.0)


                except AssertionError:
                    pass
            except AssertionError:
                pass

            
            try:
                assert el.isinsideBox and not self.elem[el.P1].isinsideBox
                try:
                    try:
                        assert el.P1 > 0
                        scaleFact = (self.elem[el.P1].x-self.elem[el.P2].x)/(el.LT+el.LP1+el.LP2)
                    except AssertionError:
                        scaleFact = (self.elem[el.P1].x-self.elem[el.P2].x)/abs(
                            self.elem[el.P1].x-self.elem[el.P2].x)

                    throatStart = self.elem[el.P2].x + el.LP2*scaleFact
                    throatEnd = throatStart + el.LT*scaleFact
                    bdr = self.xstart if (self.elem[el.P1].x < self.xstart) else self.xend           

                    cond2a = (el.P1 < 1)
                    cond2b = (not cond2a) & (throatEnd > self.xstart) & (throatEnd < self.xend)
                    cond2c = (not cond2a) & (not cond2b) & (throatStart > self.xstart) & (
                        throatStart < self.xend)
                    cond2d = not (cond2a | cond2b | cond2c)

                    try:
                        assert (cond2a | cond2c | cond2d)
                        setattr(el, "LP1mod", 0.0)
                    except AssertionError:
                        if cond2b: setattr(el, "LP1mod", (bdr-throatEnd)/scaleFact)
                
                    if cond2c: setattr(el, "LTmod", (bdr-throatStart)/scaleFact)
                    if cond2d: setattr(el, "LTmod", 0.0)


                except AssertionError:
                    pass
            except AssertionError:
                pass


def __elementList1__(self):
        _elem = np.zeros(self.nPores+self.nThroats+2, dtype=object)
        elemPore = []
        elemThroat = []
        _elem[-1] = Inlet()     # inlet = -1
        _elem[0] = Outlet(L=self.Lnetwork)     # outlet = 0
        _elem[-1].neighbours = []
        _elem[0].neighbours = []

        _numT = -1          #index of the element in the triangle array
        for p in self.pore:
            pp = Pore(p, self.poreCon[p[0]], self.throatCon[p[0]]+self.nPores)
            pp.isinsideBox = (pp.x >= self.xstart) & (pp.x <= self.xend)
            
            try:
                assert pp.G <= self.bndG1
                el = Element(Triangle(pp), self)
                _numT += 1
                el.halfAng = self.halfAngles[_numT]
            except AssertionError:
                try:
                    assert pp.G > self.bndG2
                    el = Element(Circle(pp), self)
                except AssertionError:
                    el = Element(Square(pp), self)

            el.connected = self.connected[el.indexOren]
            _elem[el.indexOren] = el
            elemPore.append(el)

        for t in self.throat:
            tt = Throat(t, self.nPores, self)
            tt.isinsideBox = (_elem[tt.P1].isinsideBox | _elem[tt.P2].isinsideBox)
            
            try:
                assert tt.G <= self.bndG1
                el = Element(Triangle(tt), self)
                _numT += 1
                el.halfAng = self.halfAngles[_numT]
                #elemTriangle.append(el)
            except AssertionError:
                try:
                    assert tt.G > self.bndG2
                    el = Element(Circle(tt), self)
                    #elemCircle.append(el)
                except AssertionError:
                    el = Element(Square(tt), self)
                    #elemSquare.append(el)

            el.connected = self.connected[el.indexOren]
            _elem[el.indexOren] = el
            elemThroat.append(el)

            try:
                assert el.conToInlet
                _elem[-1].neighbours.append(el.indexOren)
            except AssertionError:
                pass
                try:
                    assert el.conToOutlet
                    _elem[0].neighbours.append(el.indexOren)
                except AssertionError:
                    pass

        self.elemPore = np.array(elemPore)
        self.elemThroat = np.array(elemThroat)
        self.elem = np.array(_elem)

def updateConfig(input_data):
    config.title = input_data.network()
    config.calcBox = input_data.calcBox()

    (config.sigma, config.muw, config.munw, config.wat_resist,
     config.oil_resist, config.wat_dens, config.oil_dens) = input_data.fluid()

    config.SEED = input_data.randSeed()

    (config.m_minNumFillings, config.m_initStepSize, config.m_extrapCutBack,
     config.m_maxFillIncrease, config.m_StableFilling) =\
        input_data.satCovergence()
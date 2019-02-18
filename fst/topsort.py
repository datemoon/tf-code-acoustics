
from dfs_visit import DfsVisit

class TopOrderVisitor(object):
    def __init__(self, order, acyclic):
        self._order = order # it's list
        self._acyclic = acyclic # it's bool

    def InitVisit(self, fst):
        self._acyclic = True

    def InitState(self, stateid1, stateid2):
        return True

    def TreeArc(self, stateid, arc):
        return True

    def BackArc(self, stateid, arc):
        self._acyclic = False
        return True

    def ForwardOrCrossArc(self, stateid, arc):
        return True

    def FinishState(self, stateid1, stateid2, arc):
        self._order.append(stateid1)

    def FinishVisit(self):
        self._order.reverse()

def TopSort(fst):
    top_order_visitor = TopOrderVisitor(list(), acyclic = True)
    DfsVisit(fst, top_order_visitor)

    return top_order_visitor._acyclic



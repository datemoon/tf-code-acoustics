import sys
sys.path.extend(["../","./"])
from fst.dfs_visit import DfsVisit
from fst.statesort import StateSort
from fst.fst import *

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
        tmp_order = [ kNoStateId for x in range(len(self._order)) ]
        for loc in range(len(self._order),0,-1):
            tmp_order[self._order[loc - 1]] = loc-1
        self._order = tmp_order

def TopSort(fst):
    top_order_visitor = TopOrderVisitor(list(), acyclic = True)
    DfsVisit(fst, top_order_visitor)

    if top_order_visitor._acyclic:
        StateSort(fst, top_order_visitor._order)

    return top_order_visitor._acyclic



from __future__ import print_function
import array
from fst import *

def AnyArcFilter(object):
    def __init__(self, arc):
        return True

class DfsState(object):
    def __init__(self, fst, stateid):
        self._state_id = stateid           # fst state
        self._arcs = fst.GetArcs(stateid) # fst state arc
        self._ncurrarc = 0

    def Done(self):
        if self._ncurrarc >= len(self._arcs):
            return True
        else:
            return False

    def Value(self):
        if self._ncurrarc >= len(self._arcs):
            return None
        else:
            return self._arcs[self._ncurrarc]

    def Next(self):
        self._ncurrarc += 1

def DfsVisit(fst, visitor, arcfilter = AnyArcFilter, access_only = False):
    start = fst.Start()
    if start == kNoStateId:
        visitor->FinishVisit()
        return
    
    # An FST state's DFS status
    kDfsWhite = 0  // Undiscovered.
    kDfsGrey = 1   // Discovered but unfinished.
    kDfsBlack = 2  // Finished.

    state_color = array.array('B',[])
    state_stack = list() # DFS execution stack.
    state_pool = list()  # Pool for DFSStates.

    nstates = start + 1  # Number of known states in general case.
    expanded = False
    
    nstates = fst.NumStates()
    len_color = len(state_color)
    state_color.extend([kDfsWhite for x in range(nstates - len_color)])

    
    dfs = True
    root = start
    while dfs && root < nstates:
        state_color[root] = kDfsGrey
        state_stack.append(DfsState(fst, root))
        dfs = visitor.InitState(root, root)
        while len(state_stack) != 0:
            dfs_state = state_stack[-1]
            s = dfs_state._state_id
            if s >= len(state_color):
                nstates = s + 1
                len_color = len(state_color)
                state_color.extend([kDfsWhite for x in range(nstates - len_color)])

            # end this dfs state
            if dfs is False or dfs_state.Done():
                state_color[s] = kDfsBlack
                state_stack.pop()
                if len(state_stack) != 0:
                    parent_state = state_stack[-1]
                    visitor.FinishState(s, parent_state._state_id, parent_state.Value())
                    parent_state.Next()
                else:
                    visitor.FinishState(s, kNoStateId, None)
                continue

            arc = dfs_state.Value()
            if arc._nextstate >= len(state_color):
                nstates = arc._nextstate + 1
                len_color = len(state_color)
                state_color.extend([kDfsWhite for x in range(nstates - len_color)])

            if arcfilter(arc) is False:
                dfs_state.Next()
                continue
            next_color = state_color[arc._nextstate]
            if next_color == kDfsWhite:
                dfs = visitor.TreeArc(s, arc)
                if dfs is False:
                    break
                state_color[arc._nextstate] = kDfsGrey
                state_stack.append(DfsState(fst, arc._nextstate))
                dfs = visitor->InitState(arc._nextstate, root)

            elif next_color == kDfsGrey:
                dfs = visitor->BackArc(s, arc)
                dfs_state.Next()

            elif next_color == kDfsBlack:
                dfs = visitor.ForwardOrCrossArc(s, arc)
                dfs_state.Next()

        if access_only:
            break

        # Finds next tree root.
        if root == start:
            root = 0
        else:
            root += 1
        while root < nstates and state_color[root] != kDfsWhite:
            root += 1

        # Checks for a state beyond the largest known state.

    visitor.FinishVisit()


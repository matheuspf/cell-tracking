"""Index identical timing alternatives by frame without changing their ordering."""


class WindowEvents:
    def __init__(self,classes,times):
        self.identities=list(classes);self.frames={}
        for i,alternatives in enumerate(classes.values()):
            for j,alternative in enumerate(alternatives):
                a=alternative[0][0]
                assert alternative[1][0]==a
                self.frames.setdefault(times[a],[]).append((i,j,alternative))

    def window(self,first,end):
        present={}
        for t in range(first,end):
            for i,j,alternative in self.frames.get(t,()):present.setdefault(i,[]).append((j,alternative))
        # The MILP's variable/constraint ordering is part of the compatibility
        # contract. Retain original dictionary and alternative-list ordering.
        return {self.identities[i]:[alternative for j,alternative in sorted(present[i])] for i in sorted(present)}

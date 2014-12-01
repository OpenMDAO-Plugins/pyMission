# SNOPThistoryReader
# John Hwang
# June 8, 2013
# Tool for processing SNOPT_history.out files
# Useful for viewing a clean output file and plotting convergence histories

import numpy


class SNOPThistoryReader(object):

    def __init__(self, filename, constrained=True):
        f = open(filename, 'r')
        good = False
        self.raw = []
        for line in f.readlines():
            if line == '\n' or line == '1\n':
                good = False
            if good:
                self.raw.append(line[:-2])
            if line[:20] == '   Itns Major Minors':
                good = True
        f.close()

        lengths = [7, 6, 7, 8, 7, 9, 9, 15, 8, 6, 7, 8, 8]
        self.array = numpy.zeros((len(self.raw),13 if constrained else 12))
        for i in range(len(self.raw)):
            line = self.raw[i]
            for j in range(len(lengths)):
                entry = line[sum(lengths[:j]):sum(lengths[:j+1])]
                entry = entry.replace('(',' ').replace(')',' ').replace(' ','')
                if not entry == '':
                    self.array[i,j] = float(entry)

    def printRaw(self):
        for line in self.raw:
            print line

    def getDict(self):
        names = ['Total minors', 'Major', 'Minors', 'Step', \
                     'FuncEvals', 'Feasibility', 'Optimality', 'MeritFunction', \
                     'L+U', 'Bswap', 'Superbasics', 'CondReducedH', 'Penalty']
        hist = {}
        for j in range(self.array.shape[1]):
            hist[names[j]] = self.array[:,j]
        return hist


if __name__=="__main__":

    filename = 'SNOPT_print.out'

    hist = SNOPThistoryReader(filename).getDict()
    print hist['Optimality']
    print hist['Feasibility']

    SNOPThistoryReader(filename).printRaw()

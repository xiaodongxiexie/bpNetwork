#coding: utf-8


#pybrain源代码中，对于训练集测试集提供了随机打乱按比例分隔的方法，但是有时候我们想不打乱按比例分隔，此时可以对源代码部分进行修改
#主要是在datasets下的supervised，和sequential下增加一个自定义代码，主要是参考splitWithProportion，将打乱改为顺序切割
#其实可以直接在classification下增加就不用在父类上增加定义了，不过为了防止后来的引用，我还是在父类上分别增加了。

####################################################################
#SupervisedDataSet增加定义
def splitNotRandom(self, proportion=.3):
    '''define by myself for ordial seq'''

    #indicies = random.permutation(len(self))
    indicies = range(len(self))
    separator = int(len(self) * proportion)

    leftIndicies = indicies[:separator]
    rightIndicies = indicies[separator:]

    leftDs = SupervisedDataSet(inp=self['input'][leftIndicies].copy(),
                               target=self['target'][leftIndicies].copy())
    rightDs = SupervisedDataSet(inp=self['input'][rightIndicies].copy(),
                                target=self['target'][rightIndicies].copy())
    return leftDs, rightDs
    
#######################################################################
#SequentialDataSet下增加定义
def splitNotRandom(self, proportion=.3):
    '''define by myself for ordial seq'''
    l = self.getNumSequences()
    #leftIndices = sample(list(range(l)), int(l * proportion))
    leftIndices = list(range(l))[int(l * proportion):]
    leftDs = self.copy()
    leftDs.clear()
    rightDs = leftDs.copy()
    index = 0
    for seq in iter(self):
        if index in leftIndices:
            leftDs.newSequence()
            for sp in seq:
                leftDs.addSample(*sp)
        else:
            rightDs.newSequence()
            for sp in seq:
                rightDs.addSample(*sp)
        index += 1
    return leftDs, rightDs

import math
def simulatedAnnealing(rels):
    # when largest rel, possibility = 1
    T = rels[-1] - rels[0] # initial temperature
    # when smallest rel, possibility = 0.1
    Tend = -T/math.log(0.1) # final temperature
    Tgap = (T-Tend)/4
    result = []
    currentLoss = rels[0]
    for rel in rels:
        p = math.exp(-(rel-currentLoss)/T)
        T = T - Tgap
        print(p)
def linearlyPossibility(rels):
    T = rels[-1] - rels[0]
    possibilityGap = (1 - 0.1)/T #startPossibility - endPossibility
    for rel in rels:
        p = 1 - (rel - rels[0])*possibilityGap
        print(p)
simulatedAnnealing([1.111111111, 1.111111112, 1.111111112, 1.111111112, 1.111111115])
linearlyPossibility([1.111111111, 1.111111112, 1.111111112, 1.111111112, 1.111111115])
simulatedAnnealing([1.111111111, 1.4999999, 1.4999999, 1.4999999, 1.5])
linearlyPossibility([1.111111111, 1.4999999, 1.4999999, 1.4999999, 1.5])
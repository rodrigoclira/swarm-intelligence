def betterthan(fitnessA, fitnessB, maximization=False, minimization=False):
    "Tests if fitnessA is better than fitnessB according being a minimization or maximization problem"

    if (maximization and minimization) or ( not maximization and  not minimization): 
        raise ValueError
    
    if minimization:
        return fitnessA < fitnessB
    elif maximization:
        return fitnessA > fitnessB
    else:
        raise ValueError



if __name__ == '__main__':
    print(betterthan(2,3, maximization=True))
    print(betterthan(2,3, minimization=True))
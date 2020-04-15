############################
# Takes 3 integer arguments and outputs the mathematical solution of the first 
# operated on by the second
############################

from PN import predictNumber

def solveEquation(first,sign,second):
    final = 0
    first_No = predictNumber(first)
    second_No = predictNumber(second)
    
    if(sign=='plus'):
        final = first_No+second_No
    if(sign=='minus'):
        final = first_No-second_No
    return first_No,second_No,final
import sys
import typing as t

import numpy
import scipy

print(sys.version)

def function1(aaaa, bbb=4):
	"""function that rutrn a prime number
	
	Args:
	    aaaa (str): table
	    bbb (int, optional): nsk
	
	Returns:
	    int: prime number
	"""
	if aaaa > 1:
		raise ValueError(' your value of aa : ')
	return 1


def function2(aaaa:str, bbb:int=4) -> int:
	"""Summary
	
	Args:
	    aaaa (str): Description
	    bbb (int, optional): Description
	
	Returns:
	    int: Description
	"""
	return 1

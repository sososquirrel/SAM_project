import pytest
from pysquall.gg import function1


def test_function1():
	aa = 0.5
	with pytest.raises(ValueError) as e:
		function1(aa)
	assert ' your value of aa : ' in str(e)
	

	aa = 0.4
	assert function1(aa) == 1

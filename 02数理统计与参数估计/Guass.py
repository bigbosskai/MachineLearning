import math
def Guass(x,u,sigma):
	a=1.0/((2*math.pi)**0.5*sigma)
	b=math.exp(-(x-u)/(2*sigma**2))
	return a*b

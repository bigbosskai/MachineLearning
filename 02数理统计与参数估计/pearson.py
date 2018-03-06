def pearson(a,b):
	if len(a)!=len(b):
		return None
	sum=0.0
	for i in a:
		sum=sum+i
	a_mean=sum/len(a)
	sum=0.0
	for j in b:
		sum=sum+j
	b_mean=sum/len(b)
	xysum=0.0
	for idx in range(len(a)):
		xysum=xysum+a[idx]*b[idx]
	EAB=xysum/len(a)
	UP=EAB-a_mean*b_mean
	a2_sum=0.0
	b2_sum=0.0
	for i in range(len(a)):
		a2_sum=a2_sum+(a[i]-a_mean)**2
		b2_sum=b2_sum+(b[i]-b_mean)**2
	var_a=a2_sum/len(a)
	var_b=b2_sum/len(b)
	DOWN=(var_a*var_b)**0.5
	return UP/DOWN
if __name__=="__main__":
	a=[1.0,2.0,3.0,4.0,5.0]
	b=[0,0,2000.3,3.3,400.0]
	print(pearson(a,b))
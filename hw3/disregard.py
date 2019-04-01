import numpy
def to_one_hot(y, num_class=10):
    vect = []
    for i in range(len(y)):
       index = y[i]
       to_add = [0]*index + [1] + [0]*(num_class - index - 1)
       vect += [to_add]
    return numpy.array(vect)

if __name__ == '__main__':
	y = [0,1,2,3,4,5,6,7,8,9]
	print(to_one_hot(y))

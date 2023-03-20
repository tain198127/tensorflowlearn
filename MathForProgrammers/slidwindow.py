from collections import deque
array =[1,3,5,2,7,4,5,6,7]
w=3
qmax = deque()
for i in range(len(array)):
    while len(qmax) != 0 and qmax.index(len(qmax)) <= array[i]:
        qmax.pop()

    print(i,array[i])
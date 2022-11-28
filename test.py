import numpy as np

def softmax(data):
    e = np.exp(data)
    return e / sum(e)
    
arr = [1,2,3,4,5,6,7,8,9,10]

print(softmax(arr))
print(sum(softmax(arr)))

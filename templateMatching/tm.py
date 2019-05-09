import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

P = 60

def _debug_print(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


def exhaustive_search(test, reference):
    refx = reference.shape[0]
    refy = reference.shape[1]
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ref_padded = np.pad(reference, ((0, test.shape[0] - refx), (0, test.shape[1] - refy)),\
                        'constant', constant_values=((0,0),(0,0)))
    
    c = np.real(np.fft.ifft2((np.conj(np.fft.fft2(ref_padded))*np.fft.fft2(test))/\
                             np.absolute(np.conj(np.fft.fft2(ref_padded))*np.fft.fft2(test))))
    #print(test, reference)
    temp =  np.unravel_index(np.argmax(c, axis=None), c.shape)
    #print(temp)
    return int(temp[0] + refx/2), int(temp[1] + refy/2)

def exhaustive_search2(test, reference):
    refx = reference.shape[0]
    refy = reference.shape[1]
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    c = np.zeros((test.shape[0] - refx + 1, test.shape[1] - refy + 1))
    for i in range(test.shape[0] - refx + 1):
        for j in range(test.shape[1] - refy + 1):
            c[i,j] = np.sum(reference.astype(int) * test[i:i+refx, j:j+refy].astype(int))/\
                        (np.linalg.norm(reference) * np.linalg.norm(test[i:i+refx, j:j+refy]))
    #print(test, reference)
    temp =  np.unravel_index(np.argmax(c, axis=None), c.shape)
    return int(temp[0] + refx/2), int(temp[1] + refy/2)


def hierarchical_search(frame, ref):
    if ref.shape[0] <= 8 or ref.shape[1] <= 8:
        return exhaustive_search2(frame, ref)
    new_frame = cv2.pyrDown(frame)
    new_ref = cv2.pyrDown(ref)
    #print(frame.shape, ref.shape, new_frame.shape, new_ref.shape)
    x, y = hierarchical_search(new_frame, new_ref)
    #print(x, y)
    best = -math.inf
    for i in range(2 * x - 1, 2 * x + 2):
        for j in range(2 * y - 1, 2 * y + 2):
            ii = i - int(ref.shape[0] / 2)
            jj = j - int(ref.shape[1] / 2)
            if ii >= 0 and jj >= 0 and ii + ref.shape[0] <= frame.shape[0]\
                and jj + ref.shape[1] <= frame.shape[1]:
                temp = np.sum(ref.astype(int) * frame[ii:ii+ref.shape[0], jj:jj+ref.shape[1]].astype(int))/\
                        (np.linalg.norm(ref) * np.linalg.norm(frame[ii:ii+ref.shape[0], jj:jj+ref.shape[1]]))
                if temp > best:
                    best = temp
                    argbest = i, j
    return argbest[0], argbest[1]


def logarithmic_search(frame, ref):
    l = int(frame.shape[1]/4)
    x, y = int(frame.shape[0]/2), int(frame.shape[1]/2)
    
    best = -math.inf
    while(True):
        for i in range(-1, 2):
            for j in range(-1, 2):
                #print('p1 ',x+i*l, y+j*l)
                ii = x + i * l - int(ref.shape[0] / 2)
                jj = y + j * l - int(ref.shape[1] / 2)
                if ii >= 0 and jj >= 0 and ii + ref.shape[0] <= frame.shape[0]\
                    and jj + ref.shape[1] <= frame.shape[1]:
                    #print(ii,jj)
                    temp = np.sum(ref.astype(int) * frame[ii:ii+ref.shape[0], jj:jj+ref.shape[1]].astype(int))/\
                        (np.linalg.norm(ref) * np.linalg.norm(frame[ii:ii+ref.shape[0], jj:jj+ref.shape[1]]))
                    if temp > best:
                        best = temp
                        argbest = i, j
        x, y = x + argbest[0] * l, y + argbest[1] * l
        l = int(l / 2)
        if l < 1: break
    return x + argbest[0] * l * 2, y + argbest[1] * l * 2
    
    

def search(frame, ref, x, y, p, method):
    threshold = lambda x : 0 if x < 0 else x
    xt, yt = method(frame[threshold(x - p): threshold(x + p), threshold(y - p): threshold(y + p)], ref)
    return threshold(x - p) + xt, threshold(y - p) + yt

'''
test = cv2.imread('test.jpg')
ref = cv2.imread('ref.jpg')

print(exhaustive_search(test, ref))
x, y = logarithmic_search(test, ref)
print(x, y)
frame = cv2.rectangle(test,(int(y  - ref.shape[1]/2), int(x - ref.shape[0]/2)), \
                          (int(y + ref.shape[1]/2), int(x + ref.shape[0]/2)), (0, 0, 255), 3)
_debug_print(frame)
'''

cap = cv2.VideoCapture('movie.mov')
ref = cv2.imread('reference.jpg')

out = cv2.VideoWriter('output.mov', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS),\
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    exit(0)


ret, frame = cap.read()
while not ret: ret, frame = cap.read()

x, y = exhaustive_search(frame, ref)

num = 1
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        x, y = search(frame, ref, x, y, P, logarithmic_search)
        num+=1
        frame = cv2.rectangle(frame,(int(y  - ref.shape[1]/2), int(x - ref.shape[0]/2)), \
                          (int(y + ref.shape[1]/2), int(x + ref.shape[0]/2)), (0, 0, 255), 3)
        #_debug_print(frame)
        out.write(frame)
        #break
    else: break


cap.release()
out.release()

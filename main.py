import numpy as np
from typing import List
import logging as log
from PIL import Image

from math import *


class Alignement:
    def __init__(self) -> None:
        pass

    def load_img(self):
        size = 40, 40
        im1 = Image.open(r"c1.jpg").convert('L')
        im2 = Image.open(r"c2.jpg").convert('L') 

        im1.thumbnail(size, Image.Resampling.LANCZOS) 
        im2.thumbnail(size, Image.Resampling.LANCZOS) 

        # im1.show()
        # print(np.array(im1))

        al = self.get_alignement_sequence(np.array(im1), np.array(im2))

        PIL_image = Image.fromarray(np.uint8(al)).convert('L')
        PIL_image.show()
        PIL_image.save('aligned.png')

        # print(al)
    
    def dtw(self, s: list, r: list):
        """
        s and r are already array of columns"""
        n, m = len(s), len(r)
        dtw_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dtw_matrix[i, j] = np.inf
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n):
            for j in range(1, m):
                cost = self.compute_cost1(s[i], r[j])
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min
        return dtw_matrix

    def get_path(self, dtw_matrix):

        print(dtw_matrix)
        
        j = len(dtw_matrix[0]) -1
        i = len(dtw_matrix) -1
        path = [(i, j)]
        while(not(i==0 and j ==0)):
            print(i, j)
            print(path)
            a = (dtw_matrix[i-1, j], i-1, j)
            b = (dtw_matrix[i, j-1], i, j-1)
            c = (dtw_matrix[i-1, j-1], i-1, j-1)
            d = [c, a, b]
            def get_element_cost(elem: tuple):
                return elem[0]
            
            d = sorted(d, key=get_element_cost)
            print("d = ", d)
            f = d[0]
            s = d[1]
            if(f[0] == s[0]):
                if(((s[1] == i-1) and (s[2] == j-1)) ):
                    f=s
            i = f[1]
            j = f[2]
            print("f = ", f)
            path.append( (i, j ) )
        print(path)
        return path
    
    def get_alignement_sequence(self, s, r)->List:
        ma = self.dtw(np.transpose(np.array(s)), np.transpose(np.array(r)))

        pa = self.get_path(ma)
        pa.reverse()
        # paa = pa[1:]
        # pa = [(i-1, j-1) for (i, j) in paa]

        # log.getLogger().setLevel(log.INFO)
        #log.debug ("pa reversed" )
        al = np.zeros((len(s), len(r[0])))
        print(al)

        ns = np.array(s)

        for j in range(len(r[0])):
            for p in pa:
                # p = (sourcei, referencei)
                cols, colr = p
                if colr == j:
                    al[:, j] = ns[:, cols]
                    break
        return al



    def compute_cost1(self, a: list, b: list) -> int:
        """
            a et b sont des colonnes. Ils contiennent des pixels
        """
        # difference de hauteurs des colonnes avec des pixels pleins
        aa = [x for x in a if x!=255]
        bb = [x for x in b if x!=255]
        return abs(len(aa) - len(bb))
    
    def compute_cost2(self, a, b) -> int:
        """
            a et b sont des colonnes. Ils contiennent des pixels
        """
        # difference de hauteurs des colonnes avec des pixels pleins
        
        return round(abs(a-b), 1)



# a = Alignement()
# x = [
#     [1,2,3],
#     [0,0,0],
#     [1,7,8]
# ]
# y = [
#     [2,2,2,3,4],
#     [6,2,0,3,4],
#     [2,0,2,3,4],
#     [2,0,2,3,1],
# ]

# # u = [0, 3.4, 4.1, 1.2, 0.2, 0.1, 0.1]
# # v = [0.5,4,3.3,4,4.3,1.7,0.4,0.3,0.2,0.1]
# m = a.dtw(v, u)
# print(m)
# pa = a.get_path(m)
# pa.reverse()
# # paa = pa[1:]
# # pa = [(i-1, j-1) for (i, j) in paa]
# print(pa)
# # a.load_img()



a = Alignement()
a.load_img()
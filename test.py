import unittest
from main import *
class TestAlignement(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.a = Alignement()

    # def test_dtw(self):
    #     s = [
    #         [0,0,5,3],
    #         [2,5,0,0],
    #         [8,8,0,9]
    #     ]

    #     r = [
    #         [0,0,5,3],
    #         [2,5,0,0],
    #         [8,8,0,9]
    #     ]
    #     m = self.a.dtw(s, r)
    #     print(m)

    # def test_dtw2(self):
    #     s = [
    #         [0,0,],
    #         [2,5,],
    #     ]
    #     r = [
    #         [0,0,5,3],
    #         [2,5,0,0],
    #         [8,8,0,9]
    #     ]
    #     m = self.a.dtw(s, r)
    #     print('m 2')
    #     print(m)
    #     print("path2 = ")
    #     print(self.a.get_path(m))

    # def test_alignement(self):
    #     s = [
    #         [0,1,],
    #         [2,5,],
    #     ]
    #     r = [
    #         [0,0,5,3],
    #         [2,5,0,0],
    #         [8,8,0,9]
    #     ]
    #     al = self.a.get_alignement_sequence(s,r)
    #     print(al)
    
    def test_image(self):
        self.a.load_img()
import numpy as np
from typing import List
import logging as log
from PIL import Image

from math import *


class Alignement:
    def __init__(self) -> None:
        pass

    def load_img(self, source_img: str, reference_img: str):
        """
            Charge les images, fait la conversion en niveaux de gris et redimensionne les images
        """

        img_size = 80, 80
        self.source_img = Image.open(f"car_images/{source_img}.jpg").convert('L')
        self.reference_img = Image.open(f"car_images/{reference_img}.jpg").convert('L')

        self.source_img.thumbnail(img_size, Image.Resampling.LANCZOS)
        self.reference_img.thumbnail((120, 80), Image.Resampling.LANCZOS)

    def execute_alignement(self, source_img: str, reference_img: str):
        """
            Execute la fonction pour charger l'image source et l'image de reference et la fonction de l'alignement.
            Aussi, cree une nouvelle image et l'enregistre
        """

        self.load_img(source_img, reference_img)
        result_image: numpy.ndarray = self.get_alignement_sequence(np.array(self.source_img), np.array(self.reference_img))
        result_image_interpolation = self.interpolation(np.array(self.source_img), np.array(self.reference_img))

        PIL_image = Image.fromarray(np.uint8(result_image)).convert('L')
        PIL_image.show("dtw")

        # PIL_image_interpol = Image.fromarray(np.uint8(result_image_interpolation)).convert('L')
        # PIL_image_interpol.show("interpolation")

        # PIL_image_interpol = Image.fromarray(np.uint8(self.source_img)).convert('L')
        # PIL_image_interpol.show("source")

        # PIL_image_interpol = Image.fromarray(np.uint8(self.reference_img)).convert('L')
        # PIL_image_interpol.show("reference")

        
        # PIL_image.save(f"results/{source_img}-{reference_img}.png")
        # PIL_image_interpol.save(f"results/{source_img}-{reference_img}-interpol.png")

    
    def dtw(self, first_sequence: np.ndarray, second_sequence: np.ndarray) -> np.ndarray:
        """
            Crée la matrice des distances à partir de deux sequences
        """

        len_first_seq, len_second_seq = len(first_sequence), len(second_sequence)

        dtw_matrix: numpy.ndarray = np.zeros((len_first_seq, len_second_seq))
        
        for i in range(len_first_seq):
            for j in range(len_second_seq):
                dtw_matrix[i, j] = np.inf

        dtw_matrix[0, 0] = 0
        
        for i in range(1, len_first_seq):
            for j in range(1, len_second_seq):
                cost: int = self.cost(first_sequence[i], second_sequence[j])
                last_min: int = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix

    def interpolation(self, image_source: np.ndarray, image_reference: np.ndarray):
        result_matrix: np.ndarray = np.zeros((len(image_source), len(image_reference[0])))
        s, r = np.transpose(np.array(image_source)), np.transpose(np.array(image_reference))
        len_s, len_r = len(s), len(r)

        for i in range(len_r):
            j = floor(i*len_s/len_r)
            result_matrix[:, i] = image_source[:, j]

        return result_matrix

    def get_path(self, dtw_matrix: np.ndarray) -> list:
        """
            Construit le chemin avec les points minimals de la matrice de distances
        """

        columns_index: int = len(dtw_matrix[0]) - 1
        rows_index: int = len(dtw_matrix) - 1
        path: list = [(rows_index, columns_index)]

        while(not(rows_index == 0 and columns_index == 0)):
            above_value: tuple = (dtw_matrix[rows_index-1, columns_index], rows_index-1, columns_index) # (cost, row_index, col_index)
            left_value: tuple = (dtw_matrix[rows_index, columns_index-1], rows_index, columns_index-1)
            diagonal_value: tuple = (dtw_matrix[rows_index-1, columns_index-1], rows_index-1, columns_index-1)
            values_list: list = [above_value, diagonal_value, left_value]

            def get_element_cost(elem: tuple):
                return elem[0]
            
            values_list: list = sorted(values_list, key=get_element_cost)

            smaller_value: tuple = values_list[0]
            sec_smaller_value: tuple = values_list[1]

            if(smaller_value[0] == sec_smaller_value[0]): # if two 1st and 2nd values are same
                if(((sec_smaller_value[1] == rows_index-1) and (sec_smaller_value[2] == columns_index-1))): # we choose the one in the diagonal
                    smaller_value = sec_smaller_value

            rows_index = smaller_value[1]
            columns_index = smaller_value[2]
            path.append((rows_index, columns_index))

        return path
    
    def get_alignement_sequence(self, image_source: np.ndarray, image_reference: np.ndarray) -> np.ndarray:
        """
            Effectue l'alignement entre l'image source et l'image de référence
        """

        dtw_matrix: numpy.ndarray = self.dtw(np.transpose(np.array(image_source)), np.transpose(np.array(image_reference)))
        path: list = self.get_path(dtw_matrix)
        path.reverse()

        result_matrix: numpy.ndarray = np.zeros((len(image_source), len(image_reference[0])))
        image_source = np.array(image_source)

        for j in range(len(image_reference[0])):
            for path_value in path:
                source_column, reference_column = path_value
                if reference_column == j: # we found amatch
                    result_matrix[:, j] = image_source[:, source_column] # copy the column source_column innto result matrix at j
                    break

        return result_matrix


    def cost(self, column_u: np.ndarray, column_v: np.ndarray) -> int:
        """
            Calcule la distance entre deux colonnes.
            Pour l'image source: column_u et l'image de reference: column_v.
            La distance calculée ici est la différence de taille des chaque colonne.
            La hauteur représente le nombre maximum de pixel non vide (!=255 car 255 est considéré comme arriere plan)
        """

        first_match_u: numpy.ndarray = np.where(column_u != 255)[0]
        last_match_u: numpy.ndarray = np.where(reversed(column_u) != 255)[0]
        first_match_v: numpy.ndarray = np.where(column_v != 255)[0]
        last_match_v: numpy.ndarray = np.where(reversed(column_v) != 255)[0]

        first_match_u = first_match_u[0] if len(first_match_u) != 0 else 0
        last_match_u = last_match_u[0] if len(last_match_u) != 0 else 0
        first_match_v = first_match_v[0] if len(first_match_v) != 0 else 0
        last_match_v = last_match_v[0] if len(last_match_v) != 0 else 0
        
        return abs(((len(column_u) - last_match_u) - first_match_u) - ((len(column_v) - last_match_v) - first_match_v))
        

    def correlation(self, column_u: np.ndarray, column_v: np.ndarray) -> int:
        result = column_u[:, np.newaxis] * column_v
        len_u, len_v = len(column_u), len(column_v)
        total_sum = np.sum(result)
        total = total_sum / (len_u * len_v)
        # print('result',total_sum, round(total))
        return round(total)


name1 = "car1"
name2 = "car4"
a = Alignement()
a.execute_alignement(name1, name2)
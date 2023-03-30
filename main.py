import numpy as np
from typing import List
import logging as log
from PIL import Image
import cv2

from math import *

import os
import matplotlib.pyplot as plt

class Alignement:
    def __init__(self) -> None:
        pass

    def load_img(self, source_img_path: str, reference_img_path: str, is_color=False):
        """
            Charge les images, fait la conversion en niveaux de gris et redimensionne les images
        """

        DIM_SOURCE = 40, 40
        DIM_REF = 90, 40

        # load images
        source_img = cv2.imread(f"{source_img_path}")
        ref_img = cv2.imread(f"{reference_img_path}")

        # resize images
        self.source_img_o = cv2.resize(source_img, DIM_SOURCE, interpolation = cv2.INTER_AREA)
        self.reference_img_o = cv2.resize(ref_img, DIM_REF, interpolation = cv2.INTER_AREA)

        self.source_img_arr = np.array(self.source_img_o)
        self.reference_img_arr = np.array(self.reference_img_o)
        self.is_color : bool = is_color

        source_img_gray = cv2.cvtColor(self.source_img_o, cv2.COLOR_BGR2GRAY)
        reference_img_gray = cv2.cvtColor(self.reference_img_o, cv2.COLOR_BGR2GRAY)


        # put colored images into bin
        if is_color:
            _ , binary_img_s = cv2.threshold(source_img_gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.source_img_arr = np.array(binary_img_s)

            _ , binary_img_r = cv2.threshold(reference_img_gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.reference_img_arr = np.array(binary_img_r)
        else:
            self.source_img_arr = np.array(source_img_gray)
            self.reference_img_arr = np.array(reference_img_gray)

            
        # cv2.imshow('Otsu Threshold', thresh1)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return self.source_img_arr, self.reference_img_arr
    
    def gen_result_matrix(self, s: np.ndarray, r) -> np.ndarray:
        w, h = (len(s), len(r[0]))
        return np.zeros((w, h, 3), np.uint8)
    

    def execute_alignement(self, source_img_path: str, reference_img_path: str, is_color: bool):
        """
            Execute la fonction pour charger l'image source et l'image de reference et la fonction de l'alignement.
            Aussi, cree une nouvelle image et l'enregistre
        """

        
        source_img_np, ref_img_np = self.load_img(source_img_path, reference_img_path, is_color)


        result_image_dtw_height_cost: np.ndarray = self.get_dtw_alignement_sequence(source_img_np, ref_img_np, self.cost_height_diff)
        result_image_dtw_correlation_cost: np.ndarray = self.get_dtw_alignement_sequence(source_img_np, ref_img_np, self.cost_correlation)
        result_image_interpolation : np.ndarray= self.get_interpolation_alignement(source_img_np, ref_img_np)

        # PIL_image = Image.fromarray(np.uint8(result_image_dtw_height_cost)).convert('L')
        # PIL_image.show("dtw")

        empty_img = np.zeros_like(source_img_np)

        # result_concat = cv2.hconcat([result_image_dtw_height_cost, result_image_dtw_correlation_cost, result_image_interpolation])
        # original_concat = cv2.hconcat([empty_img, empty_img, empty_img])

        # all_concat = cv2.vconcat([original_concat, result_concat])


        fig, axs = plt.subplots(2, 3, figsize=(8, 8))

        # Afficher les images dans la grille
        axs[0, 0].imshow(self.source_img_o)
        axs[0, 0].set_title('source')

        axs[0, 1].imshow(self.reference_img_o)
        axs[0, 1].set_title('reference')

        axs[1, 0].imshow(result_image_dtw_height_cost)
        axs[1, 0].set_title('dtw: longueur diff')

        axs[1, 1].imshow(result_image_dtw_correlation_cost)
        axs[1, 1].set_title('dtw: correlation')

        axs[1, 2].imshow(result_image_interpolation)
        axs[1, 2].set_title('interpolation')


        # image = np.random.rand(100, 100)

        # Afficher l'image avec Matplotlib
        # plt.imshow(image, cmap='gray')
        # plt.show()
        s_name = source_img_path.split('/')[1].split('.')[0]
        r_name = reference_img_path.split('/')[1].split('.')[0]
        if not os.path.exists('resultats_all'):
            os.makedirs('resultats_all')
        plt.savefig(f'resultats_all/{s_name}_{r_name}')
        # cv2.imshow('Résultats', all_concat)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        # PIL_image_interpol = Image.fromarray(np.uint8(result_image_interpolation)).convert('L')
        # PIL_image_interpol.show("interpolation")

        
        # PIL_image.save(f"results/{source_img}-{reference_img}-dtw-hauteur.png")
        # PIL_image_interpol.save(f"results/{source_img}-{reference_img}-interpol.png")

    
    def dtw(self, first_sequence: np.ndarray, second_sequence: np.ndarray, cost_function: "function") -> np.ndarray:
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
                cost: int = cost_function(first_sequence[i], second_sequence[j])
                last_min: int = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix

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
    
    def get_interpolation_alignement(self, image_source: np.ndarray, image_reference: np.ndarray):
        s, r = np.transpose(image_source), np.transpose(image_reference)
        len_s, len_r = len(s), len(r)

        result_matrix = self.gen_result_matrix(image_source, image_reference)
        image_source_o_arr = np.array(self.source_img_o)

        for i in range(len_r):
            j = floor(i*len_s/len_r)
            result_matrix[:, i] = image_source_o_arr[:, j]

        return result_matrix

    def get_dtw_alignement_sequence(self, image_source: np.ndarray, image_reference: np.ndarray, cost: "function") -> list:
        """
            Effectue l'alignement entre l'image source et l'image de référence
        """
        transpose_source, transpose_ref = np.transpose(np.array(image_source)), np.transpose(np.array(image_reference))
        dtw_matrix: numpy.ndarray = self.dtw(transpose_source, transpose_ref, cost)
        path: list = self.get_path(dtw_matrix)
        path.reverse()
        
        result_matrix = self.gen_result_matrix(image_source, image_reference)

        image_source_o_arr = np.array(self.source_img_o)
        # image_source = np.array(image_source)

        for j in range(len(image_reference[0])):
            for path_value in path:
                source_column, reference_column = path_value
                if reference_column == j: # we found amatch
                    # print((image_source[:, source_column]))
                    result_matrix[:, j] = image_source_o_arr[:, source_column] # copy the column source_column innto result matrix at j
                    break

        return result_matrix

    def cost_height_diff(self, column_u: np.ndarray, column_v: np.ndarray) -> int:
        """
            Calcule la distance entre deux colonnes.
            Pour l'image source: column_u et l'image de reference: column_v.
            La distance calculée ici est la différence de taille des chaque colonne.
            La hauteur représente le nombre maximum de pixel non vide (!=255 car 255 est considéré comme arriere plan)
        """

        first_match_u: numpy.ndarray = np.where(column_u != 255)[0]
        last_match_u: numpy.ndarray = np.where(column_u[::-1] != 255)[0]
        first_match_v: numpy.ndarray = np.where(column_v != 255)[0]
        last_match_v: numpy.ndarray = np.where(column_v[::-1] != 255)[0]

        first_match_u = first_match_u[0] if len(first_match_u) != 0 else 0
        last_match_u = last_match_u[0] if len(last_match_u) != 0 else 0
        first_match_v = first_match_v[0] if len(first_match_v) != 0 else 0
        last_match_v = last_match_v[0] if len(last_match_v) != 0 else 0
        
        return abs(((len(column_u) - last_match_u) - first_match_u) - ((len(column_v) - last_match_v) - first_match_v))
        
    def cost_correlation(self, column_u: np.ndarray, column_v: np.ndarray) -> int:
        product_u_v = np.multiply(column_u, column_v)
        total = np.sum(product_u_v) / (len(column_u))
        return round(total)


    def main(self):

        colored_images = "colored_images" # remplacer par le chemin du dossier à lister
        bin_images = "bin_images"
        
        folders = [colored_images]
        for folder in folders:
            files = os.listdir(folder)
            # Affichage des noms de fichiers un par un
            for (i, file1)  in enumerate(files):
                for file2 in files[i+1:]:
                    print(file1, file2)
                    a.execute_alignement(f"{folder}/{file1}", f"{folder}/{file2}", is_color=(folder=="colored_images"))

name1 = "car1"
name2 = "car3"
a = Alignement()
a.main()
# a.execute_alignement(name1, name2)
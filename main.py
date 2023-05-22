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

        HEIGHT = 45
        # scale_percent = 60 # percent of original size

        scale_percent = int(100* HEIGHT / source_img.shape[0])
        width_s = int(source_img.shape[1] * scale_percent / 100)
        height_s = int(source_img.shape[0] * scale_percent / 100)
        dim_s = (width_s, HEIGHT)

        # print(source_img.shape, dim_s )

        scale_percent = int(100* HEIGHT/ ref_img.shape[0])
        width_r = int(ref_img.shape[1] * scale_percent / 100)
        height_r = int(ref_img.shape[0] * scale_percent / 100)
        dim_r = (width_r, HEIGHT)

        # print(ref_img.shape, dim_r )


        # resize images
        self.source_img_o = cv2.resize(source_img, dim_s, interpolation = cv2.INTER_AREA)
        self.reference_img_o = cv2.resize(ref_img, dim_r, interpolation = cv2.INTER_AREA)

        # self.source_img_o = cv2.resize(source_img, DIM_SOURCE, interpolation = cv2.INTER_AREA)
        # self.reference_img_o = cv2.resize(ref_img, DIM_REF, interpolation = cv2.INTER_AREA)

        source_img_arr = np.array(self.source_img_o)
        reference_img_arr = np.array(self.reference_img_o)
        self.is_color : bool = is_color

        source_img_gray = cv2.cvtColor(self.source_img_o, cv2.COLOR_BGR2GRAY)
        reference_img_gray = cv2.cvtColor(self.reference_img_o, cv2.COLOR_BGR2GRAY)


        # put colored images into bin
        if is_color:
            _ , binary_img_s = cv2.threshold(source_img_gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            source_img_arr = np.array(binary_img_s)

            _ , binary_img_r = cv2.threshold(reference_img_gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            reference_img_arr = np.array(binary_img_r)
        else:
            source_img_arr = np.array(source_img_gray)
            reference_img_arr = np.array(reference_img_gray)

            
        # cv2.imshow('Otsu Threshold', thresh1)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return source_img_arr, reference_img_arr
    
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

        # result_concat = cv2.hconcat([result_image_dtw_height_cost, result_image_dtw_correlation_cost, result_image_interpolation])
        # original_concat = cv2.hconcat([empty_img, empty_img, empty_img])

        # all_concat = cv2.vconcat([original_concat, result_concat])


        fig, axs = plt.subplots(2, 3, figsize=(8, 8))

        # Afficher les images dans la grille
        axs[0, 0].imshow(self.source_img_o)
        axs[0, 0].set_title('source')

        axs[0, 1].imshow(self.reference_img_o)
        axs[0, 1].set_title('reference')

        axs[0, 2].remove()

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
        results_folder = f"results/{'color' if is_color else 'binari'}"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        plt.savefig(f'{results_folder}/{s_name}_{r_name}')
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
    
    def get_interpolation_alignement(self, image_source_np: np.ndarray, image_reference_np: np.ndarray):
        s, r = np.transpose(image_source_np), np.transpose(image_reference_np)
        len_s, len_r = len(s), len(r)

        result_matrix = self.gen_result_matrix(image_source_np, image_reference_np)
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
        dtw_matrix: np.ndarray = self.dtw(transpose_source, transpose_ref, cost)
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


def generetate_aligned_image():

    colored_images = "colored_images" # remplacer par le chemin du dossier à lister
    bin_images = "bin_images"
    a = Alignement()
    
    folders = [bin_images, colored_images]
    for folder in folders:
        files = os.listdir(folder)
        # Affichage des noms de fichiers un par un
        for (i, file1)  in enumerate(files):
            for file2 in files[i+1:]:
                print(file1, file2)
                a.execute_alignement(f"{folder}/{file1}", f"{folder}/{file2}", is_color=(folder==colored_images))

# main()


class ImageData:
    def __init__(self, n, c, d) -> None:
        """ n = chemin vers l'image
            c = la classe de l'image
            d = la distance finaelemnt calculee par dtw avec limga de test
        """
        self.n = n
        self.c = c
        self.d = d
    
    def __str__(self) -> str:
        return f"{self.c}, {self.d}"

class KNN:
    def __init__(self, k = 3) -> None:
        self.k = k

        data_directory = "knn/classes"
        classes = os.listdir(data_directory)

        dataset = []
        for c in classes:
            images = os.listdir(f"{data_directory}/{c}")
            for i in images:
                dataset.append( ImageData( f"{data_directory}/{c}/{i}", c, 0 ))
        
        # print(dataset)

        test_directory = "knn/test"
        test_images = os.listdir(test_directory)

        a = Alignement()
        self.stat = {
            "all": 0,
            "true": 0
        }


        for ti in test_images:
            image_de_test = f"{test_directory}/{ti}"
            self.stat['all'] +=1
            self.test_image(image_de_test, dataset, a)
        
        print(f"ratio = {self.stat['true']}/ { self.stat['all']} = {(self.stat['true'] / self.stat['all'])*100 } %")
            


    def test_image(self, image: str, dataset, a,  c = None):
        image_de_test = image
        for image_de_donne in dataset:
            if image_de_test.split('/')[-1] == image_de_donne.n.split('/')[-1]:
                c = image_de_donne.c
                continue
            source_img_np, ref_img_np = a.load_img(image_de_test, image_de_donne.n, False)
            # print(image_de_test, 'et ', image_de_donne.n, end=" --> ")
            transpose_source, transpose_ref = np.transpose(np.array(source_img_np)), np.transpose(np.array(ref_img_np))
            dist = a.dtw(transpose_source, transpose_ref, a.cost_height_diff)[-1][-1]
            image_de_donne.d = dist
            # print(dist)


        dataset.sort(key=lambda x: x.d)
        # print([str(item) for item in dataset])
        
        thedist = dict()
        for data in dataset[:self.k]:
            thedist[data.c] = thedist.get(data.c, 0) + 1

        thedist = sorted(thedist.items(), key=lambda item: item[1])

        # print(thedist)
        # nous avons trié du plus peti au plus grans. Ce qui nous interesse c'est la plus grande distance donc le dernier
        predicted_class = thedist[-1][0]
        is_correct_class = predicted_class == c
        self.stat['true'] +=1 if is_correct_class else 0

        print(f"{image_de_test} ---> {predicted_class} / {c} : {is_correct_class} \n\n ")


class KMEANS:
    def __init__(self, k = 2) -> None:
        self.k = 2

        data_directory = "kmeans"

        all_images = os.listdir(data_directory)
        # print(all_images)
        dataset = []

        a = Alignement()
        for source in all_images:
            dist = 0
            for ref in all_images:
                source_img_np, ref_img_np = a.load_img(f"{data_directory}/{source}", f"{data_directory}/{ref}", False)
                transpose_source, transpose_ref = np.transpose(np.array(source_img_np)), np.transpose(np.array(ref_img_np))
                dist = dist + a.dtw(transpose_source, transpose_ref, a.cost_height_diff)[-1][-1]

            data = dict()
            data = {
                "image": source,
                "desc": dist/len(all_images),
                "cluster": None
            }

            dataset.append(data)
        # print(dataset)
        noyau = []
        ma = max(dataset, key=lambda x: x['desc'])['desc']
        mi = min(dataset, key=lambda x: x['desc'])['desc']
        pas = (ma-mi) / (k-1)
        for i in range(0, k):
            noyau.append((i*pas) + mi )
        # print(noyau)

        j = 0
        while(True):
            j+=1
            # print(j)
            clusters = []
            for n in noyau:
                clusters.append([])
            
            for image in dataset:
                image_cluster = 0
                dist_mini = abs(image['desc'] - noyau[image_cluster])
                for i, n in enumerate(noyau):
                    if abs(image['desc'] - n) < dist_mini:
                        image_cluster = i
                        dist_mini = abs(image['desc'] - n)
                image['cluster'] = image_cluster
                clusters[image_cluster].append(image)
            new_kernel = []
            for cluster in clusters:
                new_kernel.append (sum(map(lambda x: x['desc'], cluster) )/ len(cluster))

            if(new_kernel == noyau):
                break 

            noyau = [x for x in new_kernel]

        
        # print(dataset)
        # print(clusters)

        for i, cluster in enumerate(clusters):
            print('cluster ', i)
            for image in cluster:
                print(f"\t{image['image']}")
        

while True:
    i = int(input('1- Alignement \n2- KNN \n3- KMEANS\n'))
    if i == 1:
        generetate_aligned_image()
    if i == 2:
        knn = KNN(3)
    if i == 3:
        kmeans = KMEANS(2)
import pygame
import numpy as np
import math
from joblib import load
###########
#Constants#
###########

WIDTH = 420
HEIGHT = 420
COLS = 28
ROWS = 28
GRID_SIZE = COLS*ROWS
BLACK = (0,)*3
WHITE = (255,)*3
PERCENTAGE_OF_AXIS = 0.8


class Grid():

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros(GRID_SIZE)
        self.box_width = width/COLS
        self.box_height = height/ROWS
        self.box_draw_width = math.ceil(width/COLS)
        self.box_draw_height = math.ceil(height/ROWS)
        self.mlp_classifier = load("MLPclassifier.joblib")
        #self.mlp_classifier1 = load("MLPclassifierWithNormalization255.joblib")

    def draw(self, window) -> None:
        window.fill(WHITE)

        # draw boxes
        for index, value in enumerate(self.grid):
            if value == 0:
                continue
            y = (index//ROWS) * self.box_height
            x = (index % COLS) * self.box_width

            pygame.draw.rect(
                window, (255-value,)*3, (x, y, self.box_draw_width, self.box_draw_height))

        pygame.display.update()

    def hit(self, mouse_pos) -> None:
        indexs = self.__check_click(mouse_pos)
        for i, index in enumerate(indexs):

            if i != 0:
                self.grid[index] = min(self.grid[index] + 10, 255)
            else:
                self.grid[index] = 255

    def clean(self, mouse_pos) -> None:
        indexs = self.__check_click(mouse_pos)
        for i in indexs:
            self.grid[i] = 0

    def clean_all(self) -> None:
        self.grid = np.zeros(GRID_SIZE)

    def predict(self) -> None:
        out = self.mlp_classifier.predict([self.grid])
        #out1 = self.mlp_classifier1.predict([self.grid/255.0])
        #self.mlp_classifier.partial_fit([self.grid], out)
        print(out)

    def __check_click(self, mouse_pos):
        mouse_x, mouse_y = mouse_pos
        index_col = mouse_x//self.box_width
        index_row = mouse_y//self.box_height
        index = int(index_col + index_row*ROWS)
        if index < 0 or index >= GRID_SIZE:
            return []
        axis = (self.box_width*PERCENTAGE_OF_AXIS/2,
                self.box_height*PERCENTAGE_OF_AXIS/2)
        center = (index_col*self.box_width+self.box_width/2,
                  index_row*self.box_height+self.box_height/2)
        if self.__is_in_ellipse(mouse_pos, center, axis):
            return [index]
        res = [index] + \
            self.__get_adj_pos(mouse_pos, center, axis, index_col, index_row)

        return [index] + self.__get_adj_pos(mouse_pos, center, axis, index_col, index_row)

    def __is_in_ellipse(self, mouse_pos, center_pos, axis):
        return sum([pow((mouse_pos[i]-center_pos[i])/axis[i], 2) for i in range(2)]) <= 1

    def __get_adj_pos(self, mouse_pos, center_pos, axis, index_col, index_row):
        res = []
        a, b = axis
        x, y = mouse_pos
        cx, cy = center_pos

        # left part
        if x < cx-b:
            if y < cy-a:  # top part
                res.append(self.__map_2d_to_1d(index_col, index_row-1))  # top
                res.append(self.__map_2d_to_1d(
                    index_col-1, index_row-1))  # top left
            else:  # bottom part
                res.append(self.__map_2d_to_1d(
                    index_col-1, index_row+1))  # bottom left
                res.append(self.__map_2d_to_1d(
                    index_col, index_row+1))  # bottom
            res.append(self.__map_2d_to_1d(index_col-1, index_row))  # left

        # right part
        elif x > cx+b:
            if y < cy-a:  # top part
                res.append(self.__map_2d_to_1d(index_col, index_row-1))  # top
                res.append(self.__map_2d_to_1d(
                    index_col+1, index_row-1))  # top right
            else:  # bottom part
                res.append(self.__map_2d_to_1d(
                    index_col+1, index_row+1))  # bottom right
                res.append(self.__map_2d_to_1d(
                    index_col, index_row+1))  # bottom
            res.append(self.__map_2d_to_1d(index_col+1, index_row))  # right
        else:
            if y < cy-a:  # top part
                res.append(self.__map_2d_to_1d(index_col, index_row-1))  # top
            else:  # bottom part
                res.append(self.__map_2d_to_1d(
                    index_col, index_row+1))  # bottom

        return list(filter(lambda elem: elem >= 0 and elem < GRID_SIZE, res))

    def __map_2d_to_1d(self, x, y):
        return int(x+y*ROWS)


def main():
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    grid = Grid(WIDTH, HEIGHT)
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                grid.clean_all()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                grid.predict()
            if pygame.mouse.get_pressed()[0]:
                grid.hit(pygame.mouse.get_pos())
            if pygame.mouse.get_pressed()[2]:
                grid.clean(pygame.mouse.get_pos())
        grid.draw(window)
    pygame.quit()


main()


"""
TODO:
-acrescentar popup a dizer o n]umero que foi predict
-treinar com o input do usser e atualizar o classifier
-dar display de um score
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 12:58:51 2025

@author: n.zhilenkova
"""
import pygame
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pygame.init()
pygame.font.init()


class Button:
    """
    Create button

    Args:
        width : Width of button
        heigth : height of button
        inactive_col : block's color under mouth cursor (the default is (255, 255, 255))
        pressed_col : TYPE, optional
            DESCRIPTION. The default is (0, 0, 0).
    """

    def __init__(
        self,
        width,
        heigth,
        inactive_col=(255, 255, 255),
        pressed_col=(0, 0, 0),
    ):
        self.w = width
        self.h = heigth
        self.inactive_col = inactive_col
        self.pressed_col = pressed_col

    def draw(
        self,
        win: pygame.surface.Surface,
        x: float,
        y: float,
        words: str,
        size: int = 40,
        color_text: tuple = (0, 0, 0),
        action=None,
    ) -> None:
        """
        Draw blocks

        Args:
            win : screen of the game
            x : x coordinate of block
            y : y coordinate of block
            words : text on block.
            size : size of font
            color_text : the color of text (the default is (0, 0, 0))
            action : action on click (the default is None)

        """

        (x_mous, y_mous) = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if x < x_mous < x + self.w and y < y_mous < y + self.h:
            pygame.draw.rect(win, self.pressed_col, (x, y, self.w, self.h), 0)
            pygame.draw.rect(win, (255, 255, 255), (x, y, self.w, self.h), 4)
            if click[0] == 1:
                action()
        else:
            pygame.draw.rect(win, self.inactive_col, (x, y, self.w, self.h), 0)
            pygame.draw.rect(win, (255, 255, 255), (x, y, self.w, self.h), 4)
        text(win, self.w / 2 + x, self.h / 2 + y, words, size, color_text)


def text(
    win: pygame.surface.Surface,
    x: float,
    y: float,
    text: str,
    size: int,
    color: tuple = (0, 0, 0),
    name_text: str = "arialblack",
) -> None:
    """
    Drow text

    Args:
        win : screen of the game
        x : x coordinate of text
        y : y coordinate of text
        text : text
        size : size of font
        color : color of font (the default is (0, 0, 0))
        name_text : font name (the default is "arialblack")

    """
    pygame.font.init()
    font = pygame.font.SysFont(name_text, size)
    title = font.render(text, False, color)
    circle(win, x, y - title.get_height() / 2, size, 2, (0, 0, 0), text)
    win.blit(title, (x - title.get_width() / 2, y - title.get_height() / 2))


def death_screen(
    window: pygame.surface.Surface, score: int, max_sc: int, next_act: tuple
) -> None:
    """
    Create death screen

    Args:
        window : screen of the game
        score : score of the game
        max_sc : max score of the all previous games
        next_act : restart game

    """
    button_start = Button(200, 40, (255, 128, 0), (255, 255, 255))
    button_exit = Button(200, 40, (255, 128, 0), (255, 255, 255))

    pygame.draw.rect(
        window, (255, 235, 165), (200, 200, 82, 120), border_radius=12
    )
    pygame.draw.rect(
        window, (102, 51, 0), (198, 198, 86, 124), 2, border_radius=12
    )

    font = pygame.font.SysFont("arialblack", 24)

    score_1 = font.render(str(score), False, (255, 255, 255))
    circle(window, 241, 227, 24, 2, (0, 0, 0), str(score))
    window.blit(score_1, (241 - score_1.get_width() / 2, 227))

    label = font.render("score", False, (255, 0, 0))
    window.blit(label, (241 - label.get_width() / 2 + 2, 200))

    label_max = font.render("best", False, (255, 0, 0))
    window.blit(label_max, (241 - label_max.get_width() / 2, 250))

    score_max = font.render(str(max_sc), False, (255, 255, 255))
    circle(window, 241, 275, 24, 2, (0, 0, 0), str(max_sc))
    window.blit(score_max, (241 - score_max.get_width() / 2, 275))

    pygame.draw.rect(window, (102, 51, 0), (137, 397, 206, 46), 3)
    button_start.draw(
        window,
        140,
        400,
        "Restart",
        24,
        color_text=(255, 255, 255),
        action=next_act,
    )

    pygame.draw.rect(window, (102, 51, 0), (137, 457, 206, 46), 3)
    button_exit.draw(window, 140, 460, "Exit", 24, color_text=(255, 255, 255))


def circle(
    window: pygame.surface.Surface,
    x: float,
    y: float,
    size: int,
    step: float,
    color: tuple,
    text: str,
) -> None:
    """
    Create frame around text
    Args:
        window : screen of the game
        x : x coordinate of button
        y : y coordinate of button
        size : size of font
        step : size of frame
        color : color of frame
        text : text that should be framed

    """

    font = pygame.font.SysFont("arialblack", size)
    label = font.render(text, False, color)
    window.blit(label, (x - label.get_width() / 2 - step, y))
    window.blit(label, (x - label.get_width() / 2, y + step))
    window.blit(label, (x - label.get_width() / 2 + step, y))
    window.blit(label, (x - label.get_width() / 2, y - step))


def best_score(score: int) -> None:
    """
    Update best score

    Args:
        score : score of the game

    """
    with open("best_score.txt", "r") as f:
        lines = f.readlines()
        nscore = lines[0].strip()
    with open("best_score.txt", "w") as f:
        if score > int(nscore):
            f.write(str(score))
        else:
            f.write(str(nscore))


def max_score() -> int:
    """
    Load the best score from previous games

    """
    with open("best_score.txt", "r") as f:
        lines = f.readlines()
        nscore = lines[0].strip()
        return int(nscore)


def score_num(window: pygame.surface.Surface, score: int) -> None:
    """
    Drow score on screen.

    Args:
        window : screen of the game
        score : score of the game

    """
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("arialblack", 50)
    label = font.render(str(score), False, (255, 255, 255))
    window.blit(label, (240 - label.get_width() / 2, 70))


def plot_seaborn(array_counter: list, array_score: list) -> None:
    """
    Create plot scores of serial games

    Args:
        array_counter : list of numbers of the games
        array_score : list of scores of the games

    """

    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13, 8))
    fit_reg = True
    ax = sns.regplot(
        x=np.array([array_counter])[0],
        y=np.array([array_score])[0],
        x_jitter=0.1,
        scatter_kws={"color": "#36688D"},
        label="Data",
        fit_reg=fit_reg,
        line_kws={"color": "#F49F05"},
    )
    # Plot the average line
    y_mean = [np.mean(array_score)] * len(array_counter)
    ax.plot(array_counter, y_mean, label="Mean", linestyle="--")
    ax.legend(loc="upper right")
    ax.set(xlabel="# games", ylabel="score")
    plt.show()

import pygame
import neat
import os
import pickle
import random

from Bird import Bird
from Pipe import Pipe
from Base import Base

pygame.font.init()
pygame.mixer.init()  

pygame.display.set_caption("Flappy Bird")

WIN_WIDTH = 400
WIN_HEIGHT = 700


BG_DAY = pygame.transform.scale_by(pygame.image.load(os.path.join("images", "bg.png")), factor=1.5)
BG_NIGHT = pygame.transform.scale_by(pygame.image.load(os.path.join("images", "bg.png")), factor=1.5)


JUMP_SOUND = pygame.mixer.Sound("sounds/jump.wav")
HIT_SOUND = pygame.mixer.Sound("sounds/hit.wav")

STAT_FONT = pygame.font.SysFont("comicsans", 25)
GAMEOVER_FONT = pygame.font.SysFont("comicsans", 50)

def draw_window(win, bird, pipes, base, score, gameover, bg_img):
    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

    base.draw(win)
    bird.draw(win)

    if gameover:
        text = GAMEOVER_FONT.render("GAMEOVER", 1, (0, 0, 0))
        win.blit(text, (WIN_WIDTH // 2 - text.get_width() // 2, WIN_HEIGHT // 2 - 50))

    pygame.display.update()


def main():
    global BG_IMG
    BG_IMG = random.choice([BG_DAY, BG_NIGHT])  

    with open("winner_genome.pkl", "rb") as f:
        winner = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, "config-feedforward.txt")

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    score = 0
    gameover = False
    bird = Bird(150, 300)

    base = Base(600)
    pipes = [Pipe(500)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()

    speed = 5
    last_speed_increase = pygame.time.get_ticks()
    speed_increase_interval = 5000 

    run = True

    while run:
        clock.tick(30)
        current_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()

        pipe_ind = 0
        if bird:
            if len(pipes) > 1 and bird.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            break

        bird.move()

        output = net.activate((bird.y,
                               abs(bird.y - pipes[pipe_ind].height),
                               abs(bird.y - pipes[pipe_ind].bottom)))

        if output[0] > 0.5:
            bird.jump()
            JUMP_SOUND.play()  

        add_pipe = False
        removed_pipes = []
        for pipe in pipes:
            if pipe.collide(bird):  
                HIT_SOUND.play()  
                gameover = True
                run = False  

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                removed_pipes.append(pipe)

            pipe.move(speed)  

        if add_pipe:
            score += 1
            pipes.append(Pipe(450))

        for rp in removed_pipes:
            pipes.remove(rp)

        if current_time - last_speed_increase > speed_increase_interval:
            speed += 1
            Pipe.reduce_gap()  
            last_speed_increase = current_time

        base.move(speed)  
        draw_window(win, bird, pipes, base, score, gameover, BG_IMG)


if __name__ == "__main__":
    main()

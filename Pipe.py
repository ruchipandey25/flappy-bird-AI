import pygame
import os
import random

PIPE_IMG = pygame.transform.scale_by(pygame.image.load(os.path.join("images", "pipe.png")), factor=1.5)

class Pipe:
    GAP = 200  
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        """Sets the height of the top and bottom pipes correctly"""
        self.height = random.randint(100, 350)  
        self.top = self.height - self.PIPE_TOP.get_height()  
        self.bottom = self.height + self.GAP  

    def move(self, speed):
        self.x -= speed

    def draw(self, win):
        """Draw both pipes"""
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        """Check if the bird collides with the pipes"""
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        return b_point or t_point  

    @classmethod
    def reduce_gap(cls):
        """Reduce the gap over time for increased difficulty"""
        if cls.GAP > 130:
            cls.GAP -= 10

import pygame

BIRD_COLOR = (255, 200, 0)  

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 15
        self.gravity = 0.5
        self.velocity = 0

    def move(self):
        self.velocity += self.gravity
        self.y += self.velocity

    def jump(self):
        self.velocity = -8

    def draw(self, screen):
        pygame.draw.circle(screen, BIRD_COLOR, (self.x, int(self.y)), self.radius)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


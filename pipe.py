import pygame
import random

PIPE_COLOR = (0, 200, 0)  # Green color for the pipes
PIPE_WIDTH = 60
PIPE_SPEED = 5
GAP_SIZE = 150  # Space between top and bottom pipes

class Pipe:
    def __init__(self, x, screen_height):
        self.x = x
        self.width = PIPE_WIDTH
        self.speed = PIPE_SPEED
        self.screen_height = screen_height

        # Randomize pipe heights
        self.top_height = random.randint(50, screen_height // 2)
        self.bottom_height = screen_height - self.top_height - GAP_SIZE

    def move(self):
        # Move the pipe to the left
        self.x -= self.speed

    def draw(self, screen):
        # Draw the top pipe
        pygame.draw.rect(screen, PIPE_COLOR, (self.x, 0, self.width, self.top_height))
        # Draw the bottom pipe
        pygame.draw.rect(
            screen,
            PIPE_COLOR,
            (self.x, self.screen_height - self.bottom_height, self.width, self.bottom_height),
        )

    def is_off_screen(self):
        # Check if the pipe is off the screen
        return self.x + self.width < 0

    def collides_with(self, bird_rect):
    # Check if the bird collides with the top or bottom pipe
        top_pipe_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        bottom_pipe_rect = pygame.Rect(
        self.x, self.screen_height - self.bottom_height, self.width, self.bottom_height
        )
        return top_pipe_rect.colliderect(bird_rect) or bottom_pipe_rect.colliderect(bird_rect)

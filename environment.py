import pygame
import sys
import numpy as np
from bird import Bird
from pipe import Pipe

class FlappyBirdEnv:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.frame_counter = 0
        self.decision_interval = 15
        pygame.init()  # Initialize pygame once
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flappy Bird - ML Agent")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Reset the game to the initial state.
        pygame.init()
        self.bird = Bird(self.screen_width // 4, self.screen_height // 2)
        self.pipes = [Pipe(self.screen_width, self.screen_height)]
        self.score = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        self.frame_counter += 1

        if self.frame_counter % self.decision_interval == 0:
            if action == 1:
                self.bird.jump()

        self.bird.move()
        for pipe in self.pipes:
            pipe.move()

        if self.pipes[-1].x < self.screen_width // 2:
            self.pipes.append(Pipe(self.screen_width, self.screen_height))

        self.pipes = [pipe for pipe in self.pipes if not pipe.is_off_screen()]

        bird_rect = self.bird.get_rect()
        if self.bird.y - self.bird.radius < 0 or self.bird.y + self.bird.radius > self.screen_height:
            self.done = True
            return self._get_state(), -100, self.done, {"pipes_passed": self.score}

        for pipe in self.pipes:
            if pipe.collides_with(bird_rect):
                self.done = True
                return self._get_state(), -100, self.done, {"pipes_passed": self.score}

        # Base reward for survival
        reward = .1

        # Reward for passing pipes
        for pipe in self.pipes:
            if pipe.x + pipe.width < self.bird.x and not hasattr(pipe, "scored"):
                self.score += 1
                pipe.scored = True
                reward += 50  # Reward for passing a pipe

        return self._get_state(), reward, self.done, {"pipes_passed": self.score}

    def _get_state(self):
        nearest_pipe = next((pipe for pipe in self.pipes if pipe.x + pipe.width > self.bird.x), None)
        if nearest_pipe:
            gap_center = nearest_pipe.top_height + (self.screen_height - nearest_pipe.bottom_height - nearest_pipe.top_height) / 2
            state = [
                #  Normalize the state values
                self.bird.y / self.screen_height, 
                np.clip(self.bird.velocity, -20, 20) / 20,  
                (nearest_pipe.x - self.bird.x) / self.screen_width,  
                nearest_pipe.top_height / self.screen_height, 
                nearest_pipe.bottom_height / self.screen_height,  
                gap_center / self.screen_height,  
            ]
        else:
            state = [self.bird.y / self.screen_height, np.clip(self.bird.velocity, -20, 20) / 20, 1.0, 0.0, 0.0, 0.5]

        return np.array(state, dtype=np.float32)

    def render(self):
        self.screen.fill((255, 255, 255))  
        self.bird.draw(self.screen)
        for pipe in self.pipes:
            pipe.draw(self.screen)

        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()  
        self.clock.tick(60)  
        

    def close(self):
        pygame.quit()
        sys.exit()

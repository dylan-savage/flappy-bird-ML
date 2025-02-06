import pygame
import sys
from bird import Bird
from pipe import Pipe

# Initialize pygame
pygame.init()

# Constants for the game
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Clock to control frame rate
clock = pygame.time.Clock()

# Font for displaying the score
font = pygame.font.Font(None, 36)


def main():
    bird = Bird(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
    pipes = [Pipe(SCREEN_WIDTH, SCREEN_HEIGHT)]  # Start with one pipe
    score = 0  # Initialize score

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.jump()

        # Update game state
        bird.move()
        for pipe in pipes:
            pipe.move()

        # Add a new pipe when the last one is halfway across the screen
        if pipes[-1].x < SCREEN_WIDTH // 2:
            pipes.append(Pipe(SCREEN_WIDTH, SCREEN_HEIGHT))

        # Remove pipes that are off-screen
        pipes = [pipe for pipe in pipes if not pipe.is_off_screen()]

        # Check for collisions
        bird_rect = bird.get_rect()
        if bird.y - bird.radius < 0 or bird.y + bird.radius > SCREEN_HEIGHT:
            print("Game Over: Bird hit the ground or flew too high!")
            running = False
        for pipe in pipes:
            if pipe.collides_with(bird_rect):
                print("Game Over: Bird hit a pipe!")
                running = False

        # Update score: Increase when the bird passes a pipe
        for pipe in pipes:
            if pipe.x + pipe.width < bird.x and not hasattr(pipe, "scored"):
                score += 1
                pipe.scored = True  # Mark this pipe as scored

        # Draw everything
        screen.fill(WHITE)
        bird.draw(screen)
        for pipe in pipes:
            pipe.draw(screen)

        # Render and display the score
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

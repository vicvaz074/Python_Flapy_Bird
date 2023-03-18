import pygame
import sys
import random

# Inicialización
pygame.init()
clock = pygame.time.Clock()

# Configuración de la pantalla
screen_width = 500
screen_height = 700 
screen = pygame.display.set_mode((screen_width, screen_height))

# Configuración del pájaro
bird_width = 30
bird_height = 30
bird_x = screen_width // 4
bird_y = screen_height // 2
bird_color = (255, 255, 255)
bird_speed = 2 
gravity = 1

# Configuración de las tuberías
pipe_width = 80
pipe_gap = 140
pipe_color = (0, 255, 0)
pipe_speed = 3
pipes = []

# Generación de tuberías
def generate_pipes():
    pipe_height = random.randint(screen_height // 4, 3 * screen_height // 4)
    top_pipe_y = pipe_height - screen_height
    bottom_pipe_y = pipe_height + pipe_gap
    pipes.append({'top': pygame.Rect(screen_width, top_pipe_y, pipe_width, screen_height),
                  'bottom': pygame.Rect(screen_width, bottom_pipe_y, pipe_width, screen_height),
                  'passed': False})

# Dibuja tuberías
def draw_pipes():
    for pipe in pipes:
        pygame.draw.rect(screen, pipe_color, pipe['top'])
        pygame.draw.rect(screen, pipe_color, pipe['bottom'])

# Colisión
def check_collision(bird, pipes):
    for pipe in pipes:
        if bird.colliderect(pipe['top']) or bird.colliderect(pipe['bottom']):
            return True
    if bird.top <= 0 or bird.bottom >= screen_height:
        return True
    return False

# Juego principal
generate_pipes()
score = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird_speed = -13
    
    # Actualizar la posición del pájaro
    bird_speed += gravity
    bird_y += bird_speed
    bird = pygame.Rect(bird_x, bird_y, bird_width, bird_height)

    # Actualizar las tuberías
    for pipe in pipes:
        pipe['top'].left -= pipe_speed
        pipe['bottom'].left -= pipe_speed

    # Eliminar tuberías antiguas y generar nuevas
    if pipes[0]['top'].right < 0:
        pipes.pop(0)
        generate_pipes()

    # Comprobar si el pájaro ha pasado una tubería y actualizar la puntuación
    for pipe in pipes:
        if pipe['top'].right < bird_x and not pipe['passed']:
            pipe['passed'] = True
            score += 1
        if pipe['bottom'].right < bird_x and not pipe['passed']:
            pipe['passed'] = True
            score += 1

    # Dibujar
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, bird_color, bird)
    draw_pipes()

    # Mostrar puntuación
    font = pygame.font.SysFont(None, 36)
    score_text = font.render("Score: " + str(score), True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

    pygame.display.flip()

    # Colisión
    if check_collision(bird, pipes):
        break
    clock.tick(30)


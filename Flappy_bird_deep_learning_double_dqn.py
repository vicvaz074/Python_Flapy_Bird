import pygame
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

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

# Hiperparámetros
BATCH_SIZE = 32
LR = 0.001
EPSILON = 0.9
GAMMA = 0.99
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.random() < EPSILON:
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, 2)
        return action

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        samples = random.sample(self.memory, BATCH_SIZE)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*samples)

        batch_state = torch.FloatTensor(batch_state)
        batch_action = torch.LongTensor(batch_action).view(-1, 1)
        batch_reward = torch.FloatTensor(batch_reward).view(-1, 1)
        batch_next_state = torch.FloatTensor(batch_next_state)

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(-1, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
    def save(self, file_path):
        torch.save(self.eval_net.state_dict(), file_path)

    def load(self, file_path):
        self.eval_net.load_state_dict(torch.load(file_path))
        self.eval_net.eval()

class FlappyBirdEnv:
    def __init__(self):
        self.pipes = []
        self.generate_pipes()
        self.generate_pipes()
        self.reset()


    def reset(self):
        self.bird_speed = bird_speed
        self.bird_y = screen_height // 2
        self.pipe_speed = pipe_speed
        self.pipes = pipes.copy()
        self.score = 0
        return self.get_state()

    def generate_pipes(self):
        pipe_height = random.randint(screen_height // 4, 3 * screen_height // 4)
        top_pipe_y = pipe_height - screen_height
        bottom_pipe_y = pipe_height + pipe_gap
        pipes.append({'top': pygame.Rect(screen_width, top_pipe_y, pipe_width, screen_height),
                      'bottom': pygame.Rect(screen_width, bottom_pipe_y, pipe_width, screen_height),
                      'passed': False})

    def draw_pipes(self):
        for pipe in pipes:
            pygame.draw.rect(screen, pipe_color, pipe['top'])
            pygame.draw.rect(screen, pipe_color, pipe['bottom'])

    def check_collision(self, bird, pipes):
        for pipe in pipes:
            if bird.colliderect(pipe['top']) or bird.colliderect(pipe['bottom']):
                return True
        if bird.top <= 0 or bird.bottom >= screen_height:
            return True
        return False

    def get_state(self):
        closest_pipe = None
        for pipe in self.pipes:
            if pipe['top'].right >= bird_x:
                closest_pipe = pipe
                break

        if closest_pipe is None:
            return [0] * 5

        state = [
            self.bird_y / screen_height,
            self.bird_speed / 20,
            closest_pipe['top'].bottom / screen_height,
            closest_pipe['bottom'].top / screen_height,
            closest_pipe['top'].right / screen_width,
        ]
        return state

    def step(self, action):
        if action == 1:
            self.bird_speed = -13

        # Actualizar la posición del pájaro
        self.bird_speed += gravity
        self.bird_y += self.bird_speed
        bird = pygame.Rect(bird_x, self.bird_y, bird_width, bird_height)

        # Actualizar las tuberías
        for pipe in self.pipes:
            pipe['top'].left -= self.pipe_speed
            pipe['bottom'].left -= self.pipe_speed

        # Eliminar tuberías antiguas y generar nuevas
        if len(self.pipes) > 0 and self.pipes[0]['top'].right < 0:
            self.pipes.pop(0)
            self.generate_pipes()

        # Comprobar si el pájaro ha pasado una tubería y actualizar la puntuación
        for pipe in self.pipes:
            if pipe['top'].right < bird_x and not pipe['passed']:
                pipe['passed'] = True
                self.score += 1
            if pipe['bottom'].right < bird_x and not pipe['passed']:
                pipe['passed'] = True
                self.score += 1

        # Colisión
        done = self.check_collision(bird, self.pipes)
        reward = 1 if not done else -100

        return self.get_state(), reward, done, self.score
    
    def handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


def train_dqn(env, dqn, episodes):
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            dqn.store_transition(state, action, reward, next_state)
            
            env.handle_pygame_events()
            
            dqn.learn()

            episode_reward += reward
            state = next_state

            if done:
                print('Episode:', episode, 'Reward:', episode_reward)
                break

           
def main():
    env = FlappyBirdEnv()
    dqn = DQN()

    episodes = 500
    train_dqn(env, dqn, episodes)

    while True:
        state = env.reset()
        while True:
            screen.fill((0, 0, 0))
            env.draw_pipes()
            pygame.draw.rect(screen, bird_color, (bird_x, env.bird_y, bird_width, bird_height))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            action = dqn.choose_action(state)
            state, reward, done, score = env.step(action)

            font = pygame.font.SysFont(None, 36)
            score_text = font.render("Score: " + str(score), True, (255, 255, 255))
            screen.blit(score_text, (10, 10))

            pygame.display.flip()
            clock.tick(30)

            if done:
                break


if __name__ == "__main__":
    main()








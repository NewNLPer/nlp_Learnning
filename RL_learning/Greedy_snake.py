# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/25 16:54
coding with comment！！！
"""



# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/25 14:00
coding with comment！！！
"""
import pygame
import sys
import random
import math



pygame.init()

WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20
FPS = 15

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (34,139,34)
Snake_Head = (205,92,92)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("贪吃蛇")
clock = pygame.time.Clock()

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.length = 1
        self.body = [(WIDTH // 2, HEIGHT // 2)]
        self.direction = RIGHT

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0] * GRID_SIZE, head[1] + self.direction[1] * GRID_SIZE)
        self.body.insert(0, new_head)
        if len(self.body) > self.length:
            self.body.pop()
    def get_snake_len(self):
        return len(self.body)

    def change_direction(self, new_direction):
        # 防止蛇直接掉头
        if (self.direction[0] + new_direction[0], self.direction[1] + new_direction[1]) != (0, 0):
            self.direction = new_direction

    def grow(self):
        self.length += 1

    def get_head(self):
        return self.body[0]

    def get_body(self):
        return self.body

# 食物类
class Food:
    def __init__(self):
        self.position = self.generate_position()

    def generate_position(self):
        x = random.randrange(0, WIDTH // GRID_SIZE) * GRID_SIZE
        y = random.randrange(0, HEIGHT // GRID_SIZE) * GRID_SIZE
        return (x, y)

    def respawn(self):
        self.position = self.generate_position()

def draw_snake(snake):

    for i in range(len(snake.get_body())):
        if not i:
            pygame.draw.rect(screen, Snake_Head, (snake.get_body()[i][0], snake.get_body()[i][1], GRID_SIZE, GRID_SIZE))
        else:
            pygame.draw.rect(screen, GREEN, (snake.get_body()[i][0], snake.get_body()[i][1], GRID_SIZE, GRID_SIZE))


def compute_direction(point1,point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def draw_food(food):
    pygame.draw.rect(screen, RED, (food.position[0], food.position[1], GRID_SIZE, GRID_SIZE))
def step(action,snake,food):
    reward = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if action == "UP":
        snake.change_direction(UP)
    elif action == "DOWN":
        snake.change_direction(DOWN)
    elif action == "LEFT":
        snake.change_direction(LEFT)
    elif action == "RIGHT":
        snake.change_direction(RIGHT)

    old_snake_head = snake.get_head()
    snake.move()
    new_snake_head = snake.get_head()
    food_posi = food.position

    if snake.get_head() == food.position: ### 吃到食物
        snake.grow()
        food.respawn()
        reward = [True,1]
    # 检查是否发生碰撞
    elif (
        snake.get_head()[0] < 0
        or snake.get_head()[0] >= WIDTH
        or snake.get_head()[1] < 0
        or snake.get_head()[1] >= HEIGHT
        or snake.get_head() in snake.get_body()[1:]
    ): ### 撞到自己
        reward = [False,-1]
        return reward

    else: ### 仅仅是移动
        dic_1 = compute_direction(old_snake_head,food_posi)
        dic_2 = compute_direction(new_snake_head,food_posi)
        reward = [True, (dic_1 - dic_2) / 100]

    # 渲染界面
    screen.fill(WHITE)
    draw_snake(snake)
    draw_food(food)
    pygame.display.flip()
    clock.tick(FPS)

    return reward


def get_state(snake,food):

    snake_body = snake.get_body()

    snake_head = snake.get_head()

    food_pos = food.position

    state_t = [food_pos[0] - snake_head[0],food_pos[1] - snake_head[1]]

    obstacles = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0}

    x, y = snake_head

    # 检查上方是否有障碍物
    if [x, y - GRID_SIZE] in snake_body or y - GRID_SIZE < 0:
        obstacles['UP'] = 1
    # 检查下方是否有障碍物
    if [x, y + GRID_SIZE] in snake_body or y + GRID_SIZE >= WIDTH:
        obstacles['DOWN'] = 1
    # 检查左方是否有障碍物
    if [x - GRID_SIZE, y] in snake_body or x - GRID_SIZE < 0:
        obstacles['LEFT'] = 1
    # 检查右方是否有障碍物
    if [x + GRID_SIZE, y] in snake_body or x + GRID_SIZE >= WIDTH:
        obstacles['RIGHT'] = 1
    state_t += list(obstacles.values())

    return state_t



# 游戏循环
while True:
    snake = Snake()
    food = Food()
    while True:

        direction_label = {1:"UP",2:"DOWN",3:"LEFT",4:"RIGHT"}
        nums = random.randint(1,1)
        reward = step(direction_label[nums],snake,food)
        print(reward)
        print(get_state(snake,food))
        print(snake.get_head())
        if not reward[0]:
            exit()


        pygame.display.flip()
        clock.tick(FPS)

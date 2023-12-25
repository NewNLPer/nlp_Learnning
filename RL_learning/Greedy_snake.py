# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/25 14:00
coding with comment！！！
"""
import pygame
import sys
import random

# 初始化pygame
pygame.init()

# 定义常量
WIDTH, HEIGHT = 800, 800
GRID_SIZE = 20
FPS = 15

# 定义颜色
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (34,139,34)
Snake_Head = (205,92,92)

# 定义方向
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# 蛇类
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


# 初始化游戏
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("贪吃蛇")
clock = pygame.time.Clock()

snake = Snake()
food = Food()

def draw_snake(snake):

    for i in range(len(snake.get_body())):
        if not i:
            pygame.draw.rect(screen, Snake_Head, (snake.get_body()[i][0], snake.get_body()[i][1], GRID_SIZE, GRID_SIZE))
        else:
            pygame.draw.rect(screen, GREEN, (snake.get_body()[i][0], snake.get_body()[i][1], GRID_SIZE, GRID_SIZE))


def draw_food(food):
    pygame.draw.rect(screen, RED, (food.position[0], food.position[1], GRID_SIZE, GRID_SIZE))

def step(action):
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

    snake.move()
    print(snake.get_body())
    if snake.get_head() == food.position:
        snake.grow()
        food.respawn()

    # 检查是否发生碰撞
    if (
        snake.get_head()[0] < 0
        or snake.get_head()[0] >= WIDTH
        or snake.get_head()[1] < 0
        or snake.get_head()[1] >= HEIGHT
        or snake.get_head() in snake.get_body()[1:]
    ):
        print(" 撞到墙或者撞到自己 ")
        pygame.quit()
        sys.exit()

    # 渲染界面
    screen.fill(WHITE)
    draw_snake(snake)
    draw_food(food)
    pygame.display.flip()
    clock.tick(FPS)


# 游戏循环
while True:
    # 在这里调用 step 函数，传入相应的动作，比如 step("UP")
    # 可以通过键盘事件获取用户输入，或者通过神经网络等方式决定动作
    # 这里使用 pygame 的 KEYDOWN 事件作为示例，你可以根据需要修改
    dic_label = {1:"UP",2:"DOWN",3:"LEFT",4:"RIGHT"}
    nums = random.randint(1,4)
    step(dic_label[nums])


    # for event in pygame.event.get():
    #     if event.type == pygame.KEYDOWN:
    #         if event.key == pygame.K_UP:
    #             step("UP")
    #         elif event.key == pygame.K_DOWN:
    #             step("DOWN")
    #         elif event.key == pygame.K_LEFT:
    #             step("LEFT")
    #         elif event.key == pygame.K_RIGHT:
    #             step("RIGHT")

    # pygame.time.delay(10)  # 降低速度，可根据需要调整
    pygame.display.flip()
    clock.tick(FPS)

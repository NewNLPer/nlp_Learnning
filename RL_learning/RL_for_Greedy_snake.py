# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/14 20:11
coding with comment！！！
"""

"""
Speak something clearly
"""
import pygame
import random
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device) + '...')
# 初始化
pygame.init()

# 游戏参数
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 10
FPS = 20

# 颜色定义
WHITE = (255, 255, 255)
food_color = (255, 0, 0)
snake_body = (237, 145, 33)
snake_head = (255, 215, 0)
discount_factor = 0.5
play_iter = 100000

# 游戏类
class SnakeGame:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.grid_size = GRID_SIZE
        self.fps = FPS

        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("贪吃蛇游戏")

        self.snake = [(100, 100), (90, 100), (80, 100)]
        self.direction = (1, 0)  # 初始方向向右
        self.food = self.generate_food()

    def generate_food(self):
        while True:
            food = (random.randrange(0, self.width, self.grid_size),
                    random.randrange(0, self.height, self.grid_size))
            if food not in self.snake:
                return food

    def draw_grid(self):
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, WHITE, (x, 0), (x, self.height))
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, WHITE, (0, y), (self.width, y))

    def draw_snake(self):
        for i in range(len(self.snake)):
            if not i:
                pygame.draw.rect(self.screen, snake_head, (self.snake[i][0], self.snake[i][1], self.grid_size, self.grid_size))
            else:
                pygame.draw.rect(self.screen, snake_body,(self.snake[i][0], self.snake[i][1], self.grid_size, self.grid_size))
        # for segment in self.snake:
        #     pygame.draw.rect(self.screen, GREEN, (segment[0], segment[1], self.grid_size, self.grid_size))

    def draw_food(self):
        pygame.draw.rect(self.screen, food_color, (self.food[0], self.food[1], self.grid_size, self.grid_size))

    def check_collision(self):
        head = self.snake[0]
        if head in self.snake[1:] or \
                head[0] < 0 or head[0] >= self.width or \
                head[1] < 0 or head[1] >= self.height:
            return True
        return False

    def move_snake(self, action):
        if action == 1 and self.direction != (0, 1):  # 上
            self.direction = (0, -1)
        elif action == 2 and self.direction != (0, -1):  # 下
            self.direction = (0, 1)
        elif action == 3 and self.direction != (1, 0):  # 左
            self.direction = (-1, 0)
        elif action == 4 and self.direction != (-1, 0):  # 右
            self.direction = (1, 0)

        new_head = (self.snake[0][0] + self.direction[0] * self.grid_size,
                    self.snake[0][1] + self.direction[1] * self.grid_size)

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.food = self.generate_food()
        else:
            self.snake.pop()

    def step(self, action):
        reward=0

        len_ori = len(self.snake)
        self.move_snake(action)
        len_now = len(self.snake)

        if len_now - len_ori: # 吃到食物，应该得到奖励
            reward = 5
        if self.check_collision(): # 吃到自己或者碰到墙
            reward = -5
            # return False  # 游戏结束

        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.draw_snake()
        self.draw_food()

        pygame.display.flip()
        self.clock.tick(self.fps)
        # return True  # 游戏继续
        return reward

    def get_snake_state(self):

        matrix = np.zeros((40, 40)).tolist()

        for item in self.snake:

            # print([int(item[0] / 10)][int(item[1] / 10)])

            matrix[int(item[0] / 10)][int(item[1] / 10)] = 1

        matrix[int(self.food[0] / 10)][int(self.food[1] / 10)] = 2

        return matrix

class DQN_TP(nn.Module):
    def __init__(self,input_dim,out_dim):
        """
        :param input_dim: 40*40
        :param out_dim: action_choose (up 1,down 2,left 3,right 4)
        """
        super(DQN_TP, self).__init__()
        self.hidden_dim = 100
        self.action_choose = out_dim
        self.relu = nn.ReLU()
        self.get_norm = nn.Softmax()
        self.MLP_1 = nn.Linear(input_dim,self.hidden_dim)
        self.MLP_2 = nn.Linear(input_dim*self.hidden_dim,self.action_choose)

    def forward(self, state):
        """
        :param self:
        :param state:give snake state to get argmax_Q* -> action
        :return: every action to Q_value
        """
        hidden_vc = self.MLP_1(state)
        ac_hidden_vc = self.relu(hidden_vc)
        ac_hidden_vc = torch.unsqueeze(torch.flatten(ac_hidden_vc),0)
        logits = self.get_norm(self.MLP_2(ac_hidden_vc))
        return logits


if __name__ == "__main__":

    RL_model = DQN_TP(40, 4).cuda()
    optimizer = torch.optim.Adam(RL_model.parameters(), lr=0.00001)
    Loss_function = nn.MSELoss()
    for iter in tqdm(range(play_iter)):
        game = SnakeGame()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            state_t = torch.Tensor(game.get_snake_state()).cuda()

            logits_t = RL_model(state_t)

            q_t_for_a_t = float(logits_t.max())

            action_t = int(logits_t.argmax()) + 1


            reward_t = game.step(action_t)

            if reward_t == -5:
                Loss_for_RL = Loss_function(torch.Tensor([q_t_for_a_t]),torch.Tensor([reward_t + 0]))

                Loss_for_RL.requires_grad_(True)

                optimizer.zero_grad()

                Loss_for_RL.backward()

                optimizer.step()
                print("epoch : {}，loss : {}".format(iter,Loss_for_RL.item()))

                break

            state_t_1 = torch.Tensor(game.get_snake_state()).cuda()

            max_q_t_1 = float(RL_model(state_t_1).max())

            Loss_for_RL = Loss_function(torch.Tensor([q_t_for_a_t]),torch.Tensor([reward_t + discount_factor * max_q_t_1]))

            Loss_for_RL.requires_grad_(True)

            optimizer.zero_grad()

            Loss_for_RL.backward()

            optimizer.step()

            print("epoch : {}，loss : {}".format(iter,Loss_for_RL.item()))




# RL_model = DQN_TP(40,4).cuda()
#
#
#
# # def main():
#     game = SnakeGame()
#     while True:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
#
#         print("原始状态：",game.get_snake_state())
#         nums=random.randint(1,4)
#         reward=game.step(nums)
#         print("目前状态：",game.get_snake_state())
#         print("当前状态下做出action所得出的奖励：",reward)
#         exit()
        # print(st)
        # if st == -5:
        #     break


        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_UP]:
        #     action = 1
        # elif keys[pygame.K_DOWN]:
        #     action = 2
        # elif keys[pygame.K_LEFT]:
        #     action = 3
        # elif keys[pygame.K_RIGHT]:
        #     action = 4
        # else:
        #     action = 0
        #
        # if not game.step(action):
        #     print("游戏结束！")
        #     break

# if __name__ == "__main__":
#     main()




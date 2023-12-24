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
import math
import torch.utils.data as Data

class Experience_pool():
    """
    定义回收经验池，存储为（s(t),a(t),r(t),s(t+1)）
    s(t): t时刻的状态
    a(t): t时刻采取的行动
    r(t): t时刻状态下采取某个行动获取的奖励
    s(t+1): t时刻采取某个行动后发生改变的状态
    """
    def __init__(self,Storage_length):
        self.Storage_length = Storage_length
        self.Content = []
    def add_element(self,element):
        if len(self.Content) < self.Storage_length:
            self.Content.append(element)
        else:
            self.Content.pop(0)
            self.Content.append(element)
    def Randomize_Samples(self,batch_size):
        numbers = list(range(len(self.Content)))
        random.shuffle(numbers)
        sampled_numbers = numbers[:batch_size]
        need_sample = []
        for item in sampled_numbers:
            need_sample.append(self.Content[item])
        return need_sample


class Mydata(Data.Dataset):
    def __init__(self,data1):
        self.data1=data1
        self.len=len(self.data1)

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        return self.data1[item]


s=[(1,2),(2,3),(3,4)]
my_data = Mydata(s)
train_data=Mydata(s)


def my_collate(batch):
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    res=0
    for item in data:
        res=max(res,len(item))
    mask=[0]*len(data)
    for i in range(len(data)):
        mask[i]=[True]*len(data[i])+[False]*(res-len(data[i]))
        data[i]=data[i]+[0]*(res-len(data[i]))
        label[i]=label[i]+[0]*(res-len(label[i]))
    return torch.tensor(data),label,torch.tensor(mask)


train_loader=Data.DataLoader(
    dataset=train_data,
    shuffle=True,
    batch_size=128,
    collate_fn=my_collate
)
















# 初始化
pygame.init()

# 游戏参数
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
FPS = 20
ban_dic={1:(0,1),2:(0,-1),3:(1,0),4:(-1,0)}


# 颜色定义
WHITE = (255, 255, 255)
food_color = (255, 0, 0)
snake_body = (237, 145, 33)
snake_head = (255, 215, 0)
discount_factor = 0.78
play_iter = 100000

# 游戏类
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT
        self.grid_size = GRID_SIZE
        self.fps = FPS
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("贪吃蛇游戏")
        self.snake = [(100, 100), (80, 100), (80, 100)]
        nums = random.randint(1,4)
        self.direction = ban_dic[nums]  # 初始方向向右
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
        if action == 1 and self.direction != (0, 1):  # 下
            self.direction = (0, -1)
        elif action == 2 and self.direction != (0, -1):  # 上
            self.direction = (0, 1)
        elif action == 3 and self.direction != (1, 0):  # 左
            self.direction = (-1, 0)
        elif action == 4 and self.direction != (-1, 0):  # 右
            self.direction = (1, 0)
        else:
            print("=============== 出现错误方向 ==================")
            print(self.direction)
            print(action)
            print("=====================================")
            exit()
        new_head = (self.snake[0][0] + self.direction[0] * self.grid_size,
                    self.snake[0][1] + self.direction[1] * self.grid_size)

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.food = self.generate_food()
        else:
            self.snake.pop()


    def compute_direction(self,point1,point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance


    def step(self, action):
        reward=0
        old_head = self.snake[0]
        old_lens = len(self.snake)
        self.move_snake(action)
        new_head = self.snake[0]
        new_lens = len(self.snake)
        self.screen.fill((0, 0, 0))
        self.draw_grid()
        self.draw_snake()
        self.draw_food()

        pygame.display.flip()
        self.clock.tick(self.fps)

        if self.check_collision(): # 吃到自己或者碰到墙
            reward -= 2
            return [False , reward]
        elif new_lens > old_lens: # 如果吃到果实
            reward += 3
            return [True , reward]
        else: # 啥也没发生，仅仅是移动，需要考虑其余食物的距离来计算间接奖励
            old_dis = self.compute_direction(old_head,self.food)
            new_dis = self.compute_direction(new_head,self.food)
            reward += ( old_dis - new_dis) / 100
            return [True , reward]

    def get_sort_dic(self,logits):
        candidate_action = {}
        list_from_tensor = logits[0].tolist()
        for i in range(len(list_from_tensor)):
            candidate_action[i + 1] = list_from_tensor[i]
        sorted_items = sorted(candidate_action.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = dict(sorted_items)
        return sorted_dict



    def get_snake_state(self):
        """
        -1 表示墙
        0 表示空地
        1 表示蛇头
        2 表示蛇身
        3 表示食物
        :return:
        """
        init_state = [[0] * (int(WIDTH / GRID_SIZE)) for _ in range( int(WIDTH / GRID_SIZE))]
        for i in range(int(WIDTH / GRID_SIZE)):
            for j in range(int(WIDTH / GRID_SIZE)):
                if not i or not j:
                    init_state[i][j] = -1
                if int(WIDTH / GRID_SIZE) - 1 in [i,j]:
                    init_state[i][j] = -1
        for i in range(len(self.snake)):
            if not i:
                init_state[int(self.snake[i][0] / 20)][int(self.snake[i][1] / 20)] = 1
            else:
                init_state[int(self.snake[i][0] / 20)][int(self.snake[i][1] / 20)] = 2
        init_state[int(self.food[0] / 20)][int(self.food[1] / 20)] = 3
        return init_state,self.direction



class DQN_TP(nn.Module):
    def __init__(self,input_dim,out_dim):
        """
        :param input_dim: 20*20
        :param out_dim: action_choose (up 1,down 2,left 3,right 4)
        """
        super(DQN_TP, self).__init__()
        self.hidden_dim = 128
        self.action_choose = out_dim
        self.relu = nn.ReLU()
        self.MLP_1 = nn.Linear(input_dim,self.hidden_dim)
        self.MLP_2 = nn.Linear(input_dim * self.hidden_dim,self.action_choose)

    def forward(self, state):
        """
        :param self:
        :param state:give snake state to get argmax_Q* -> action
        :return: every action to Q_value
        """
        hidden_vc = self.MLP_1(state)
        ac_hidden_vc = self.relu(hidden_vc)
        ac_hidden_vc = torch.unsqueeze(torch.flatten(ac_hidden_vc),0)
        logits = self.MLP_2(ac_hidden_vc)
        return logits

"==========================================================================="
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x




if __name__ == "__main__":

    RL_model = DQN_TP(20, 4).cuda()
    optimizer = torch.optim.Adam(RL_model.parameters(), lr=0.01)
    Loss_function = nn.MSELoss()
    for iter in tqdm(range(play_iter)):
        game = SnakeGame()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            state_t,direction = game.get_snake_state()

            state_t = torch.unsqueeze(torch.Tensor(state_t),0).cuda()

            logits_t = RL_model(state_t)
            # q_t_for_a_t = float(logits_t.max())
            condidate_action = game.get_sort_dic(logits_t)
            for key in condidate_action:
                if ban_dic[key] != direction:
                    q_t_for_a_t = condidate_action[key]
                    reward_t = game.step(key)
                    break

            reward = reward_t[-1]
            if not reward_t[0]:
                Loss_for_RL = Loss_function(torch.Tensor([q_t_for_a_t]),torch.Tensor([reward - 2]))
                Loss_for_RL.requires_grad_(True)
                optimizer.zero_grad()
                Loss_for_RL.backward()
                optimizer.step()
                print("epoch : {}，loss : {} ，reward : {}".format(iter,Loss_for_RL.item(),reward))
                break
            else:
                state_t_1 = torch.unsqueeze(torch.Tensor(game.get_snake_state()[0]),0).cuda()
                max_q_t_1 = float(RL_model(state_t_1).max())
                Loss_for_RL = Loss_function(torch.Tensor([q_t_for_a_t]),torch.Tensor([reward + discount_factor * max_q_t_1]))
                Loss_for_RL.requires_grad_(True)
                optimizer.zero_grad()
                Loss_for_RL.backward()
                optimizer.step()
                print("epoch : {}，loss : {} ,reward : {}".format(iter,Loss_for_RL.item(),reward))
#



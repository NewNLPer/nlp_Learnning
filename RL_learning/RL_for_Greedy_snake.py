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


### 初始参数设置
pygame.init()
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
FPS = 20
direction_selection = {1:(1,0),2:(-1,0),3:(0,1),4:(0,-1)}
WHITE = (255, 255, 255)
food_color = (255, 0, 0)
snake_body = (237, 145, 33)
snake_head = (255, 215, 0)
discount_factor = 0.9
play_iter = 100


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
    def __len__(self):
        return len(self.Content)
    def is_full(self):
        if len(self.Content) == self.Storage_length:
            return True
        else:
            return False

class Mydata(Data.Dataset):
    def __init__(self,data1):
        self.data1=data1
        self.len=len(self.data1)

    def __len__(self):
        return self.len
    def __getitem__(self, item):
        return self.data1[item]


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
        self.snake = [(100, 100), (80, 100), (60, 100)]
        num = random.randint(2,4)
        self.direction = direction_selection[num]  # 初始方向向右
        self.food = self.generate_food()


    def generate_food(self):
        while True:
            food = (random.randrange(0, self.width , self.grid_size),
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
        if action == 1 :  # 上
            self.direction = (-1, 0)
        elif action == 2:  # 下
            self.direction = (1, 0)
        elif action == 3 :  # 左
            self.direction = (0, -1)
        elif action == 4 :  # 右
            self.direction = (0, 1)
        # else:
        #     print("Test .... 出现错误方向，当前方向 {}，action_choose {}".format(self.direction,action))
        #     exit()
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
            reward -= 3
            # print("吃到自己或者撞墙",self.snake[0])
            return [0 , reward],self.snake[0],self.food
        elif new_lens > old_lens: # 如果吃到果实
            reward += 2
            return [1 , reward],self.snake[0],self.food
        else: # 啥也没发生，仅仅是移动，需要考虑其余食物的距离来计算间接奖励
            old_dis = self.compute_direction(old_head,self.food)
            new_dis = self.compute_direction(new_head,self.food)
            reward += ( old_dis - new_dis) / 100
            return [1 , reward],self.snake[0],self.food


    def get_sort_dic(self,logits):
        candidate_action = {}
        list_from_tensor = logits[0].tolist()
        for i in range(len(list_from_tensor)):
            candidate_action[i + 1] = list_from_tensor[i]
        sorted_items = sorted(candidate_action.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = dict(sorted_items)
        return sorted_dict


    def get_know_obstacle(self,matrix,snake_head):

        if -1 in snake_head or 20 in snake_head:
            return [1,1,1,1]
        else:
            direction_state = {(-1, 0):0, (1, 0):0, (0, -1):0, (0, 1):0}
            for item in direction_state:
                direction_test_x = snake_head[0] + item[0]
                direction_test_y = snake_head[1] + item[1]
                try:
                    nums = matrix[direction_test_x][direction_test_y]
                except:
                    nums = 1
                if nums in [0,3]:
                    nums = 0
                else:
                    nums = 1
                direction_state[item] = nums
            return list(direction_state.values())


    def get_snake_state(self):
        """
        0 表示空地
        1 表示蛇头
        2 表示蛇身
        3 表示食物
        s(t) -> 横纵坐标相对于蛇头的位置，蛇头的上下左右方向是否存在障碍物。
        :return:
        """
        init_state = [[0] * (int(WIDTH / GRID_SIZE)) for _ in range( int(WIDTH / GRID_SIZE))]

        for i in range(len(self.snake)):
            if not i:
                init_state[int(self.snake[i][0] / 20)][int(self.snake[i][1] / 20)] = 1
            else:
                init_state[int(self.snake[i][0] / 20)][int(self.snake[i][1] / 20)] = 2
        init_state[int(self.food[0] / 20)][int(self.food[1] / 20)] = 3
        state_content = [self.food[0] - self.snake[0][0],self.food[1] - self.snake[0][1]]
        process_out_snake_head = (int(self.snake[0][0] / 20),int(self.snake[0][1] / 20))
        state_content += self.get_know_obstacle(init_state,process_out_snake_head)

        return state_content,init_state,process_out_snake_head




class DQN_TP(nn.Module):
    def __init__(self,input_dim,out_dim):
        """
        :param input_dim: 1*7
        :param out_dim: action_choose (up 1,down 2,left 3,right 4)
        """
        super(DQN_TP, self).__init__()
        self.hidden_dim = 128
        self.action_choose = out_dim
        self.MLP_1 = nn.Linear(input_dim,self.hidden_dim)
        self.MLP_2 = nn.Linear(self.hidden_dim,self.action_choose)
        self.relu = nn.ReLU()
        self.st = nn.Softmax()

    def forward(self, state):
        """
        :param self:
        :param state:give snake state to get argmax_Q* -> action
        :return: every action to Q_value
        """
        hidden_vc = self.MLP_1(state)
        ac_hidden_vc = self.relu(hidden_vc)
        logits = self.st(self.MLP_2(ac_hidden_vc))

        return logits

# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_dim)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


def get_choose_action(marix,snake_head):
    action_choose = [(-1,0),(1,0),(0,-1),(0,1)]
    action = []
    for i in range(len(action_choose)):
        new_x = snake_head[0] + action_choose[i][0]
        new_y = snake_head[1] + action_choose[i][1]
        try:
            nums = marix[new_x][new_y]
        except:
            nums = -1
        if nums == 2:
            continue
        else:
            action.append(i+1)
    return action


def my_collate(batch):
    q_t = [item[-1] for item in batch]
    reward = [item[2] for item in batch]
    s_t_1 = [item[3] for item in batch]


    return q_t, reward, s_t_1


if __name__ == "__main__":
    """
    1. 模型控制agent进行与环境进行交互
    2. 将交互的信息存储到经验池
    3. 随机采样经验池的信息进行训练更新模型参数
    """
    RL_model = DQN_TP(6, 4).cuda()
    train_data = Experience_pool(100)
    optimizer = torch.optim.Adam(RL_model.parameters(), lr=0.0001)
    Loss_function = nn.MSELoss()
    # start_train = 0

    for iter_ in tqdm(range(play_iter)):
        game = SnakeGame()
        with torch.no_grad():
            for ac in tqdm(range(10),desc=" Agent与环境交互收集数据 ... "):
                game = SnakeGame()
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()

                    state_t, pos_t ,snake_head_t= game.get_snake_state()
                    state_t_cuda = torch.unsqueeze(torch.Tensor(state_t), 0).cuda()
                    logits_t = RL_model(state_t_cuda)
                    condidate_action = game.get_sort_dic(logits_t)

                    Action_to_be_selected = get_choose_action(pos_t,snake_head_t)
                    for key in condidate_action:
                        if key not in Action_to_be_selected:
                            continue
                        q_t_for_a_t = condidate_action[key]
                        reward_t,head ,food= game.step(key)
                        if head[0] >= 400 or head[0] < 0 or head[1] >= 400 or head[1] < 0 :
                            state_t_1 = [food[0] - head[0],food[1] - head[1],1,1,1,1]
                            q_t_for_a_t = -1
                            break
                        else:
                            state_t_1, pos_t_1, snake_head_t_1 = game.get_snake_state()
                            break

                    train_data.add_element([state_t,key,reward_t[1],state_t_1,q_t_for_a_t])
                    # print([state_t,key,reward_t[1],state_t_1,q_t_for_a_t])
                    if not reward_t[0]:
                        break

        torch.set_grad_enabled(True)

        data = Mydata(train_data.Randomize_Samples(64))


        train_loader = Data.DataLoader(
            dataset=data,
            shuffle=True,
            batch_size=16,
            collate_fn=my_collate
        )


        for epoch in tqdm(range(1, 4),desc=" 开始更新网络参数 "):
            train_loss = 0
            for q_t, reward, s_t_1 in train_loader:
                q_t = torch.unsqueeze(torch.Tensor(q_t),1).cuda()
                reward = torch.unsqueeze(torch.Tensor(reward), 1).cuda()
                s_t_1 = torch.Tensor(s_t_1).cuda()
                logits_t_1 = RL_model(s_t_1)
                Loss_for_RL = Loss_function(q_t, discount_factor * logits_t_1 + reward)
                Loss_for_RL.requires_grad_(True)
                optimizer.zero_grad()
                Loss_for_RL.backward()
                optimizer.step()
                train_loss += Loss_for_RL.item()
            print("epoch : {}，loss : {} ".format(epoch, train_loss / 4))



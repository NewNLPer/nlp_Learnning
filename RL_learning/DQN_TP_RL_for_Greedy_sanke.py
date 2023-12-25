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
import torch.utils.data as Data
import torch
import torch.nn as nn
from tqdm import tqdm


pygame.init()

WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
FPS = 15
discount_factor = 0.9
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (34,139,34)
Snake_Head = (205,92,92)

play_iter = 100

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


class DQN_TP(nn.Module):
    def __init__(self,input_dim,out_dim):
        """
        :param input_dim: 1*6
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


def get_sort_dic(logits):
    candidate_action = {}
    list_from_tensor = logits[0].tolist()
    candidate = ["UP","DOWN","LEFT","RIGHT"]

    for i in range(len(list_from_tensor)):
        candidate_action[candidate[i]] = list_from_tensor[i]
    sorted_items = sorted(candidate_action.items(), key=lambda x: x[1], reverse=True)
    sorted_dict = dict(sorted_items)
    return sorted_dict

def my_collate(batch):
    q_t = [item[-1] for item in batch]
    reward = [item[1] for item in batch]
    s_t_1 = [item[3] for item in batch]
    return q_t, reward, s_t_1

if __name__ == "__main__":

    RL_model = DQN_TP(6, 4).cuda()
    train_data = Experience_pool(600)
    optimizer = torch.optim.Adam(RL_model.parameters(), lr=0.0001)
    Loss_function = nn.MSELoss()

    for kim in range(play_iter):
        for i in tqdm(range(1,21),desc=" Agent与环境交互收集状态数据中 "):
            snake = Snake()
            food = Food()
            symbol = 0
            while True:
                state_t = get_state(snake, food)
                state_t_cuda = torch.unsqueeze(torch.Tensor(state_t), 0).cuda()
                logits_t = RL_model(state_t_cuda)
                candidate = get_sort_dic(logits_t)
                for key in candidate:
                    symbol = 1
                    reward = step(key, snake, food)
                    q_t_for_a_t = candidate[key]
                    state_t_1 = get_state(snake, food)
                    if symbol:
                        break
                train_data.add_element([state_t ,reward[1] ,key,state_t_1 ,q_t_for_a_t])
                if not reward[0]:
                    pygame.display.flip()
                    clock.tick(FPS)
                    break


        data = Mydata(train_data.Randomize_Samples(512))
        train_loader = Data.DataLoader(
            dataset=data,
            shuffle=True,
            batch_size=64,
            collate_fn=my_collate
        )
        for epoch in tqdm(range(1, 10), desc=" DQN开始更新网络参数 "):
            train_loss = 0
            for q_t, reward, s_t_1 in train_loader:
                q_t = torch.unsqueeze(torch.Tensor(q_t), 1).cuda()
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





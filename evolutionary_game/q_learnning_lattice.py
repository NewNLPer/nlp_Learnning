# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/7 18:00
coding with comment！！！
"""

import torch
import random
import json
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("you are using {} ... ".format(device))


class Lattice():
    def __init__(self,nums):

        self.L=nums
        self.lattice_0=np.random.randint(0,3,(self.L,self.L)).tolist()
        self.lattice_1 = np.random.randint(0, 3, (self.L, self.L)).tolist()
        self.q_table_dic_0 = {str(i) +"-"+ str(j): np.random.randint(0, 1, (5, 6)) for i in range(self.L) for j in range(self.L)}
        self.q_table_dic_1 = {str(i) +"-"+ str(j): np.random.randint(0, 1, (5, 6)) for i in range(self.L) for j in range(self.L)}
        self.label_dic = {0:[-1,0],1:[1,0],2:[0,-1],3:[0,1],4:[0,0],5:[0,0]}
        self.eplison = 0.02
        self.R0 = 1
        self.R1 = 1
        self.P0 = 0.1
        self.P1 = -0.4
        self.S0 = 0
        self.S1 = 0
        self.T0 = 1.3
        self.T1 = 1.3

        """
        [上，下，左，右，复制层，不动]
        """

    def get_someone(self):
        i=random.randint(0,self.L - 1)
        j=random.randint(0,self.L - 1)
        return [i,j]
    def Boundary_treatment(self,position_1):
        if 0 <= position_1[0] <= self.L - 1 and 0 <= position_1[1] <= self.L - 1:
            return position_1
        else:
            if position_1[0] == -1:
                position_1[0] = self.L - 1

            if position_1[0] == self.L:
                position_1[0] = 0

            if position_1[1] == -1:
                position_1[1] = self.L - 1

            if position_1[1] == self.L:
                position_1[1] = 0

            return position_1

    def get_nerbio(self,position):

        nerbio=[[position[0]+1,position[1]],[position[0]-1,position[1]],[position[0],position[1]-1],[position[0],position[1]+1]]

        nerbio=[self.Boundary_treatment(item) for item in nerbio]+[position]

        return nerbio

    def count_collaborator(self,position,layer):
        nerbio=self.get_nerbio(position)
        collaborator_nums=0
        none_position=[]

        for item in nerbio[:4]:
            if layer[item[0]][item[1]]==1:
                collaborator_nums+=1
        for item in nerbio:
            if not layer[item[0]][item[1]]:
                none_position.append(item)

        return collaborator_nums,none_position



    def Determine_location(self,play_pos,ner_pos):
        # [上，下，左，右，复制层，不动]
        sure_pos=[]
        for i in range(len(play_pos)):
            sure_pos.append(play_pos[i]-ner_pos[i])
        if sure_pos in [[0,1 - self.L],[0,1]]: # 左
            return 2
        elif sure_pos in [[1-self.L,0],[1,0]]: # 上
            return 0
        elif sure_pos in [[0,self.L-1],[0,-1]]: # 右
            return 3
        elif sure_pos in [[self.L-1,0],[-1,0]]: # 下
            return 1
        elif sure_pos == [0,0]:
            return 4

    def find_max_index(self,lst):
        # 初始化最大数值和索引
        max_value = None
        max_index = None
        # 遍历列表
        for i, value in enumerate(lst):
            # 如果当前元素不是"*"且大于最大数值
            if value != "*" and (max_value is None or value > max_value):
                # 更新最大数值和索引
                max_value = value
                max_index = i
        return max_index

    def moving(self,position):

        collaborator_nums = self.count_collaborator(position,self.lattice_0)[0]
        none_position = self.count_collaborator(position,self.lattice_0)[1]
        position_q_table=self.q_table_dic_0[str(position[0])+str(position[1])]

        action_set=position_q_table[collaborator_nums]
        candidate_set = ["*"] * len(action_set)
        if not none_position:
            candidate_set[-1] = action_set[-1]
        else:
            for item in none_position:
                candidate_set[self.Determine_location(position,item)]=action_set[self.Determine_location(position,item)]
            candidate_set[-1]=action_set[-1]

        if random.randint(1,50)==1: # 随机选择
            valid_indices = [i for i, elem in enumerate(candidate_set) if elem != '*']
            random_index = random.choice(valid_indices)
        else:
            random_index = self.find_max_index(candidate_set)

        for i in range(len(position)):
            position[i]+=self.label_dic[random_index][i]
        move_postion = self.Boundary_treatment(position)

        if random_index==4:# 复制层  (另一层网络)
            self.q_table_dic_1["".join(move_postion)]=self.q_table_dic_0["".join(position)]
            self.q_table_dic_0["".join(position)]=np.random.randint(0,1,(5,6)).tolist()
            self.lattice_1[move_postion[0]][move_postion[1]]=self.lattice_0[position[0]][position[1]]
            self.lattice_0[position[0]][position[1]]=0

        elif random_index==5: # 不动  (同一层网络)
            pass
        else: # (同一层网络)
            self.q_table_dic_0["".join(move_postion)]=self.q_table_dic_0["".join(position)]
            self.q_table_dic_0["".join(position)]=np.random.randint(0,1,(5,6)).tolist()
            self.lattice_0[move_postion[0]][move_postion[1]]=self.lattice_0[position[0]][position[1]]
            self.lattice_0[position[0]][position[1]]=0


    def game(self,position):
        nerbo=self.get_nerbio(position)
        personal_fit=0

        for item in nerbo:
            if self.lattice_0[item[0]][item[1]]:
                if self.lattice_0[position[0]][position[1]]==1 and self.lattice_0[item[0]][item[1]]==1:
                    personal_fit+=self.R0
                elif self.lattice_0[position[0]][position[1]]==1 and self.lattice_0[item[0]][item[1]]==2:
                    personal_fit+=self.S0
                elif self.lattice_0[position[0]][position[1]]==2 and self.lattice_0[item[0]][item[1]]==2:
                    personal_fit+=self.P0
                elif self.lattice_0[position[0]][position[1]]==2 and self.lattice_0[item[0]][item[1]]==1:
                    personal_fit+=self.T0

        return personal_fit

    def Policy_Update(self,position):

        max_fit_police=["*",-2]

        nerbo = self.get_nerbio(position)
        for item in nerbo:
            if self.lattice_0[item[0]][item[1]]:
                nerbio_fit=self.game(item)
                if nerbio_fit > max_fit_police[-1]:
                    max_fit_police[0] = self.lattice_0[item[0]][item[1]]
                    max_fit_police[1] = nerbio_fit
        personal_fit=self.game(position)
        if personal_fit > max_fit_police[-1]:
            max_fit_police[0] = self.lattice_0[position[0]][position[1]]
            max_fit_police[1] = personal_fit
        self.lattice_0[position[0]][position[1]]=max_fit_police[0]

    def q_table_updata(self,position):
        pass










# [上，下，左，右，复制层，不动]

my_lattice=Lattice(3)
my_lattice.moving([1,2])








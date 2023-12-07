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
        self.q_table_dic = {str(i) + str(j): np.random.randint(0, 1, (5, 6)) for i in range(self.L) for j in range(self.L)}
        self.label_dic={"up":0,"down":1,"left":2,"right":3,"copy":4,"stay":5}
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

    def count_collaborator(self,position):
        nerbio=self.get_nerbio(position)
        collaborator_nums=0
        none_position=[]

        for item in nerbio[:4]:
            if self.lattice_0[item[0]][item[1]]==1:
                collaborator_nums+=1
        for item in nerbio:
            if not self.lattice_0[item[0]][item[1]]:
                none_position.append(item)

        return collaborator_nums,none_position



    def Determine_location(self,play_pos,ner_pos):
        # [上，下，左，右，复制层，不动]
        sure_pos=[]
        for i in range(len(play_pos)):
            sure_pos.append(play_pos[i]-ner_pos[i])
        if sure_pos == [0,1]:
            return 2 # 左
        elif sure_pos == [1,0]:
            return 0
        elif sure_pos == [0,-1]:
            return 3
        elif sure_pos == [-1,0]:
            return 1
        elif sure_pos == [0,0]:
            return 4



    def get_max_q_table(self,q_table):
        pass

    def moving(self,position):

        collaborator_nums = self.count_collaborator(position)[0]
        none_position = self.count_collaborator(position)[1]
        position_q_table=self.q_table_dic[str(position[0])+str(position[1])]

        action_set=position_q_table[collaborator_nums]
        print(action_set)
        if not none_position:
            exit()
        else:
            candidate_set=["*"]*len(action_set)
            print(none_position)
            for item in none_position:
                print(self.Determine_location(position,item))
                candidate_set[self.Determine_location(position,item)]=action_set[self.Determine_location(position,item)]
            candidate_set[-1]=action_set[-1]
            print(candidate_set)




my_lattice=Lattice(3)
my_lattice.moving([1,2])








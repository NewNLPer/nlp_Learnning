# -*- coding: utf-8 -*-
"""
@author: NewNLPer
@time: 2023/12/7 18:00
coding with comment！！！
"""
import random
import numpy as np
from tqdm import tqdm
import time


class Lattice():
    def __init__(self,nums):
        # [上，下，左，右，复制层，不动]
        self.L=nums
        self.q_table_dic_0 = {str(i) + "-" + str(j): np.random.randint(0, 1, (5, 6)) for i in range(self.L) for j in range(self.L)}
        self.q_table_dic_1 = {str(i) + "-" + str(j): np.random.randint(0, 1, (5, 6)) for i in range(self.L) for j in range(self.L)}
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
        self.epochs = 20000
        self.alpha = 0.1
        self.gamma = 0.9
        self.ro= 0.6
        self.lattice_0 = np.random.choice([0, 1, 2], size=(self.L, self.L), p=[0.4, 0.3, 0.3]).astype(int).tolist()
        self.lattice_1 = np.random.choice([0, 1, 2], size=(self.L, self.L), p=[0.4, 0.3, 0.3]).astype(int).tolist()

        """
        [上，下，左，右，复制层，不动]
        """

    def get_someone(self):
        i=random.randint(0,self.L - 1)
        j=random.randint(0,self.L - 1)
        return [i,j]


    def get_idex(self,position):
        return "-".join(str(item) for item in position)


    def count_c(self,state):
        lattice_sum_c=0
        if not  state:
            for item in self.lattice_0:
                lattice_sum_c+=item.count(1)
            return lattice_sum_c

        elif state:
            for item in self.lattice_1:
                lattice_sum_c+=item.count(1)
            return lattice_sum_c

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

    def count_collaborator(self,position,state):

        if not state:
            nerbio = self.get_nerbio(position)
            collaborator_nums = 0
            none_position = []
            for item in nerbio[:4]:
                if self.lattice_0[item[0]][item[1]] == 1:
                    collaborator_nums += 1
            for item in nerbio:
                if not self.lattice_0[item[0]][item[1]]:
                    none_position.append(item)
            return collaborator_nums, none_position

        elif state:
            nerbio = self.get_nerbio(position)
            collaborator_nums = 0
            none_position = []
            for item in nerbio[:4]:
                if self.lattice_1[item[0]][item[1]] == 1:
                    collaborator_nums += 1
            for item in nerbio:
                if not self.lattice_1[item[0]][item[1]]:
                    none_position.append(item)
            return collaborator_nums, none_position


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
        max_value = None
        max_index = None
        for i, value in enumerate(lst):
            if value != "*" and (max_value is None or value > max_value):
                max_value = value
                max_index = i
        return max_index

    def moving(self,position,state):

        if not state:

            collaborator_nums = self.count_collaborator(position, 0)[0]
            none_position = self.count_collaborator(position, 0)[1]
            position_q_table = self.q_table_dic_0[self.get_idex(position)]

            action_set = position_q_table[collaborator_nums]
            candidate_set = ["*"] * len(action_set)

            if not none_position:
                candidate_set[-1] = action_set[-1]
            else:
                for item in none_position:
                    candidate_set[self.Determine_location(position, item)] = action_set[self.Determine_location(position, item)]
                candidate_set[-1] = action_set[-1]

            if random.randint(1, 50) == 1:  # 随机选择
                valid_indices = [i for i, elem in enumerate(candidate_set) if elem != '*']
                random_index = random.choice(valid_indices)
            else:
                random_index = self.find_max_index(candidate_set)

            for i in range(len(position)):
                position[i] += self.label_dic[random_index][i]
            move_postion = self.Boundary_treatment(position)

            if random_index == 4:  # 复制层  (另一层网络)
                self.q_table_dic_1[self.get_idex(move_postion)] = self.q_table_dic_0[self.get_idex(position)]
                self.q_table_dic_0[self.get_idex(position)] = np.random.randint(0, 1, (5, 6)).tolist()
                self.lattice_1[move_postion[0]][move_postion[1]] = self.lattice_0[position[0]][position[1]]
                self.lattice_0[position[0]][position[1]] = 0
                new_state = self.count_collaborator(move_postion, 1)[0]
                new_state_parameters = max(self.q_table_dic_1[self.get_idex(move_postion)][new_state])


            elif random_index == 5:  # 不动  (同一层网络)
                new_state = self.count_collaborator(move_postion, 0)[0]
                new_state_parameters = max(self.q_table_dic_0[self.get_idex(move_postion)][new_state])
                pass
            else:  # (同一层网络)
                self.q_table_dic_0[self.get_idex(move_postion)] = self.q_table_dic_0[self.get_idex(position)]
                self.q_table_dic_0[self.get_idex(position)] = np.random.randint(0, 1, (5, 6)).tolist()
                self.lattice_0[move_postion[0]][move_postion[1]] = self.lattice_0[position[0]][position[1]]
                self.lattice_0[position[0]][position[1]] = 0
                new_state = self.count_collaborator(move_postion, 0)[0]
                new_state_parameters = max(self.q_table_dic_0[self.get_idex(move_postion)][new_state])

            return {"move_postion": move_postion,
                    "random_index_action": random_index,
                    "q_value": candidate_set[random_index],
                    "ori_collaborator_nums": collaborator_nums,
                    "max_q_new_state": new_state_parameters}

        elif state:

            collaborator_nums = self.count_collaborator(position, 1)[0]
            none_position = self.count_collaborator(position, 1)[1]
            position_q_table = self.q_table_dic_1[self.get_idex(position)]

            action_set = position_q_table[collaborator_nums]
            candidate_set = ["*"] * len(action_set)

            if not none_position:
                candidate_set[-1] = action_set[-1]
            else:
                for item in none_position:
                    candidate_set[self.Determine_location(position, item)] = action_set[self.Determine_location(position, item)]
                candidate_set[-1] = action_set[-1]

            if random.randint(1, 50) == 1:  # 随机选择
                valid_indices = [i for i, elem in enumerate(candidate_set) if elem != '*']
                random_index = random.choice(valid_indices)
            else:
                random_index = self.find_max_index(candidate_set)

            for i in range(len(position)):
                position[i] += self.label_dic[random_index][i]
            move_postion = self.Boundary_treatment(position)

            if random_index == 4:  # 复制层  (另一层网络)
                self.q_table_dic_0[self.get_idex(move_postion)] = self.q_table_dic_1[self.get_idex(position)]
                self.q_table_dic_1[self.get_idex(position)] = np.random.randint(0, 1, (5, 6)).tolist()
                self.lattice_0[move_postion[0]][move_postion[1]] = self.lattice_1[position[0]][position[1]]
                self.lattice_1[position[0]][position[1]] = 0
                new_state = self.count_collaborator(move_postion, 0)[0]
                new_state_parameters = max(self.q_table_dic_0[self.get_idex(move_postion)][new_state])


            elif random_index == 5:  # 不动  (同一层网络)
                new_state = self.count_collaborator(move_postion, 1)[0]
                new_state_parameters = max(self.q_table_dic_1[self.get_idex(move_postion)][new_state])
                pass
            else:  # (同一层网络)
                self.q_table_dic_1[self.get_idex(move_postion)] = self.q_table_dic_1[self.get_idex(position)]
                self.q_table_dic_1[self.get_idex(position)] = np.random.randint(0, 1, (5, 6)).tolist()
                self.lattice_1[move_postion[0]][move_postion[1]] = self.lattice_1[position[0]][position[1]]
                self.lattice_1[position[0]][position[1]] = 0
                new_state = self.count_collaborator(move_postion, 1)[0]
                new_state_parameters = max(self.q_table_dic_1[self.get_idex(move_postion)][new_state])

            return {"move_postion": move_postion,
                    "random_index_action": random_index,
                    "q_value": candidate_set[random_index],
                    "ori_collaborator_nums": collaborator_nums,
                    "max_q_new_state": new_state_parameters}

    def game(self,position,state):


        if not state:
            nerbo = self.get_nerbio(position)
            personal_fit = 0
            for item in nerbo:
                if self.lattice_0[item[0]][item[1]]:
                    if self.lattice_0[position[0]][position[1]] == 1 and self.lattice_0[item[0]][item[1]] == 1:
                        personal_fit += self.R0
                    elif self.lattice_0[position[0]][position[1]] == 1 and self.lattice_0[item[0]][item[1]] == 2:
                        personal_fit += self.S0
                    elif self.lattice_0[position[0]][position[1]] == 2 and self.lattice_0[item[0]][item[1]] == 2:
                        personal_fit += self.P0
                    elif self.lattice_0[position[0]][position[1]] == 2 and self.lattice_0[item[0]][item[1]] == 1:
                        personal_fit += self.T0
            return personal_fit

        elif state:

            nerbo = self.get_nerbio(position)
            personal_fit = 0
            for item in nerbo:
                if self.lattice_0[item[0]][item[1]]:
                    if self.lattice_0[position[0]][position[1]] == 1 and self.lattice_0[item[0]][item[1]] == 1:
                        personal_fit += self.R1
                    elif self.lattice_0[position[0]][position[1]] == 1 and self.lattice_0[item[0]][item[1]] == 2:
                        personal_fit += self.S1
                    elif self.lattice_0[position[0]][position[1]] == 2 and self.lattice_0[item[0]][item[1]] == 2:
                        personal_fit += self.P1
                    elif self.lattice_0[position[0]][position[1]] == 2 and self.lattice_0[item[0]][item[1]] == 1:
                        personal_fit += self.T1
            return personal_fit



    def Policy_Update(self,position,state):

        if not state:
            max_fit_police=["*",-999]
            nerbo = self.get_nerbio(position)
            for item in nerbo:
                if self.lattice_0[item[0]][item[1]]:
                    nerbio_fit=self.game(item,0)
                    if nerbio_fit > max_fit_police[-1]:
                        max_fit_police[0] = self.lattice_0[item[0]][item[1]]
                        max_fit_police[1] = nerbio_fit
            personal_fit=self.game(position,0)
            if personal_fit > max_fit_police[-1]:
                max_fit_police[0] = self.lattice_0[position[0]][position[1]]
                max_fit_police[1] = personal_fit

            self.lattice_0[position[0]][position[1]]=max_fit_police[0]

        elif state:
            max_fit_police=["*",-999]
            nerbo = self.get_nerbio(position)
            for item in nerbo:
                if self.lattice_1[item[0]][item[1]]:
                    nerbio_fit=self.game(item,1)
                    if nerbio_fit > max_fit_police[-1]:
                        max_fit_police[0] = self.lattice_1[item[0]][item[1]]
                        max_fit_police[1] = nerbio_fit
            personal_fit=self.game(position,1)
            if personal_fit > max_fit_police[-1]:
                max_fit_police[0] = self.lattice_1[position[0]][position[1]]
                max_fit_police[1] = personal_fit

            self.lattice_1[position[0]][position[1]]=max_fit_police[0]


    def q_table_updata(self,need_dic,personal_fit,state):
        """
{                   
                    "move_postion": move_postion,
                    "random_index_action": random_index,
                    "q_value": candidate_set[random_index],
                    "ori_collaborator_nums": collaborator_nums,
                    "max_q_new_state": new_state_parameters}
        """
        if not state:
            self.q_table_dic_0[self.get_idex(need_dic["move_postion"])][need_dic["ori_collaborator_nums"]][need_dic["random_index_action"]] = ( 1 - self.alpha ) * need_dic["q_value"] + self.alpha * (personal_fit + self.gamma * need_dic["max_q_new_state"])
        elif state:
            self.q_table_dic_1[self.get_idex(need_dic["move_postion"])][need_dic["ori_collaborator_nums"]][need_dic["random_index_action"]] = ( 1 - self.alpha ) * need_dic["q_value"] + self.alpha * (personal_fit + self.gamma * need_dic["max_q_new_state"])



    def main(self):
        fc_list=[]
        for epoch in tqdm(range(self.epochs)):
            while True:
                position = self.get_someone() # 取点

                # print([self.lattice_0[position[0]][position[1]],self.lattice_1[position[0]][position[1]]])

                if sum([self.lattice_0[position[0]][position[1]],self.lattice_1[position[0]][position[1]]]) > 0 :
                    break
            if self.lattice_0[position[0]][position[1]]:
                need_dic_0=self.moving(position,0) # 网格0移动
                benifit_0=self.game(position, 0) # 博弈
                self.Policy_Update(position,0) # 策略更新
                self.q_table_updata(need_dic_0,benifit_0,0) # q表更新

            if self.lattice_1[position[0]][position[1]]:
                need_dic_1=self.moving(position,1) # 网格1移动
                benifit_1=self.game(position, 1) # 博弈
                self.Policy_Update(position,1) # 策略更新
                self.q_table_updata(need_dic_1,benifit_1,1) # q表更新

            fc_0=self.count_c(0)

            fc_1=self.count_c(1)

            fc_list+=[[fc_0,fc_1]]

        print(fc_list)


if __name__ == "__main__":

    start_time=time.time()
    my_lattice=Lattice(100)
    my_lattice.main()
    end_time=time.time()
    print("time comsume :{}".format(end_time-start_time))



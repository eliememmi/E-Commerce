import math

import numpy as np


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        self.n = num_users
        self.m = num_arms
        self.reward_matrix = np.zeros((num_users, num_arms))
        self.apparition_matrix = np.zeros((num_users, num_arms))
        self.apparition_matrix_modulo = np.zeros((num_users, num_arms))
        self.arms_thresh = arms_thresh
        self.phase_len = phase_len
        self.means_arms = np.zeros(num_arms)
        self.T = num_rounds
        self.current_user = 0
        self.current_arm = 0
        self.count = 0
        self.count_modulo = 0
        self.to_use = -1
        self.users_distribution = users_distribution
        self.forbidden_value = []

        # TODO: Decide what/if to store. Could be used in the future
        pass

    def update_everything(self):
        if self.count_modulo % self.phase_len == 0:
            self.count_modulo = 0
            self.apparition_matrix_modulo = np.zeros((self.n, self.m))

    def choose_arm(self, user_context):
        """Initialization"""

        if self.count_modulo % self.phase_len == 0:
            self.count_modulo = 0
            self.to_use = -1
            self.apparition_matrix_modulo = np.zeros((self.n, self.m))

        """ Counting iteration """
        self.current_user = user_context
        self.count = self.count + 1
        self.count_modulo = self.count_modulo + 1

        """First Exploration"""
        for i in range(self.m):
            if self.apparition_matrix[user_context][i] < 2:
                choosen_arm = i
                self.apparition_matrix[user_context][choosen_arm] += 1
                self.apparition_matrix_modulo[user_context][choosen_arm] += 1
                self.current_arm = choosen_arm
                self.to_use = -1
                return choosen_arm

        """Is the vector armsthresh 'thin' ? See explanation in PDF"""

        if(self.phase_len - np.sum(self.arms_thresh))< 5 and self.phase_len - self.count_modulo < 25 :
            for i in range(self.n):
                a = np.sum(self.apparition_matrix_modulo[:,i])
                if self.arms_thresh[i] > a:
                    choosen_arm = i
                    #self.to_use = i
                    self.apparition_matrix[user_context][choosen_arm] += 1
                    self.apparition_matrix_modulo[user_context][choosen_arm] += 1
                    self.current_arm = choosen_arm
                    return choosen_arm

        """Choice another arm fearing that it will disappear"""
        number_to_use_without_user_context = 0

        if self.to_use != -1:
            number_to_use_without_user_context = np.sum(self.apparition_matrix_modulo[:, self.to_use])

        how_many_times_we_need_to_take_it = self.arms_thresh[self.to_use] - (number_to_use_without_user_context)

        if (self.to_use != -1):
            love_to_use = []
            for i in range(self.n):
                check_arms_less_good_than_to_use = 0
                for j in range(self.m):
                    if (self.apparition_matrix[i][self.to_use] != 0 and self.apparition_matrix[i][j] != 0):
                        if (self.reward_matrix[i][self.to_use] / self.apparition_matrix[i][self.to_use]) > (
                                self.reward_matrix[i][j] / self.apparition_matrix[i][j]):
                            check_arms_less_good_than_to_use += 1
                        if (check_arms_less_good_than_to_use == (self.m) - 1):
                            love_to_use.append(i)

            not_love_to_use = []
            for i in range(self.n):
                if i not in love_to_use:
                    not_love_to_use.append(i)

            how_many_will_take_it = 0

            for i in love_to_use:
                how_many_will_take_it += (100 - self.count_modulo) * (self.users_distribution[i])

            combien_pour_ceux_qui_aime_pas = how_many_times_we_need_to_take_it - how_many_will_take_it

        if self.to_use != -1 and (self.arms_thresh[self.to_use]) > np.sum(self.apparition_matrix_modulo, axis=0)[
            self.to_use] and 0 < ((combien_pour_ceux_qui_aime_pas / len(not_love_to_use))) and (
                self.to_use not in self.forbidden_value):
            choosen_arm = self.to_use
            self.apparition_matrix[user_context][choosen_arm] += 1
            self.apparition_matrix_modulo[user_context][choosen_arm] += 1
            self.current_arm = choosen_arm
            return choosen_arm
        elif self.to_use != -1 and self.arms_thresh[self.to_use] < np.sum(self.apparition_matrix_modulo, axis=0)[
            self.to_use]:
            self.to_use = -1

        """Default choice by the best mean reward"""
        for i in range(self.m):
            for j in range(self.m):
                if (self.reward_matrix[user_context][i] / self.apparition_matrix[user_context][i]) > (
                        self.reward_matrix[user_context][j] / self.apparition_matrix[user_context][j]):
                    if self.arms_thresh[i] > np.sum(self.apparition_matrix_modulo, axis=0)[i] and (
                            i not in self.forbidden_value):
                        choosen_arm = i
                        self.to_use = i
                        self.apparition_matrix[user_context][choosen_arm] += 1
                        self.apparition_matrix_modulo[user_context][choosen_arm] += 1
                        self.current_arm = choosen_arm
                        return choosen_arm

        choosen_arm = np.argmax(self.reward_matrix[user_context])
        self.apparition_matrix[user_context][choosen_arm] += 1
        self.apparition_matrix_modulo[user_context][choosen_arm] += 1
        self.current_arm = choosen_arm

        return choosen_arm

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """

        self.reward_matrix[self.current_user][self.current_arm] += reward

        counteur = 0
        if self.count_modulo % self.phase_len == 0:
            for j in range(self.m):
                for i in range(self.n):
                    counteur += self.apparition_matrix_modulo[i, j]
                if (counteur) < self.arms_thresh[j]:
                    self.forbidden_value.append(j)
                counteur = 0

        # TODO: Use this information for your algorithm
        pass

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_931203129_932191265"
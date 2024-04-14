import numpy as np
from simulator.constants import *
from k_means import K_Means


def normalize_qoe(qoe: float):
    # if reward_function == 'linear':
    #     if qoe <= -6.7:
    #         return -1 * 10
    #     return ((qoe - 0.3) / 7.3) * 10
    return max(qoe, -50)


class ServerBuffer:
    def __init__(self, n_agents: int, time_slot: int, device_info: list, k: int):
        self.n_agents = n_agents
        self.time_slot = time_slot
        self.k = k

        self.previous_qoe = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_bitrate = [[0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_buff_size = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_delay = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_throughput = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.r_list = []
        self.r_dump = []

        self.buffer_len = 0
        self.current_bw_co = [1 / self.n_agents for _ in range(self.n_agents)]

        self.n_types = 3
        self.device_type = device_info
        self.wss = []

    def get_latest_reward(self):
        user_avg_qoes = []
        for qoe in self.previous_qoe:
            avg_qoe = np.mean(qoe)
            user_avg_qoes.append(np.mean(avg_qoe))
        reward = np.min(user_avg_qoes)
        index = np.argmin(user_avg_qoes)
        return reward, index

    def reset_previous(self):
        self.previous_qoe = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_bitrate = [[0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_buff_size = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_delay = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]
        self.previous_throughput = [[0.0 for _ in range(self.time_slot)] for _ in range(self.n_agents)]

    def send_msg(self, qoe: float, bitrate: float, buff_size: float, delay: float, throughput: float, index: int):
        self.previous_qoe[index][0:-1] = self.previous_qoe[index][1:]
        self.previous_bitrate[index][0:-1] = self.previous_bitrate[index][1:]
        self.previous_buff_size[index][0:-1] = self.previous_buff_size[index][1:]
        self.previous_delay[index][0:-1] = self.previous_delay[index][1:]
        self.previous_throughput[index][0:-1] = self.previous_throughput[index][1:]

        self.previous_qoe[index][-1] = qoe
        self.previous_bitrate[index][-1] = bitrate
        self.previous_buff_size[index][-1] = buff_size
        self.previous_delay[index][-1] = delay
        self.previous_throughput[index][-1] = throughput

        self.buffer_len += 1

    def get_previous_state(self):
        server_state = [[] for _ in range(self.n_agents)]
        for index in range(self.n_agents):
            avg_buf = np.mean(self.previous_buff_size[index])
            avg_delay = np.mean(self.previous_delay[index])
            avg_bitrate = np.mean(self.previous_bitrate[index])
            avg_qoe = np.mean(self.previous_qoe[index])
            avg_throughput = np.mean(self.previous_throughput[index])

            server_state[index] = [
                self.device_type[index] / self.n_types,
                avg_buf / BUFFER_NORM_FACTOR,
                avg_delay / M_IN_K / BUFFER_NORM_FACTOR,
                avg_bitrate / 5,
                normalize_qoe(avg_qoe) / 10,
                avg_throughput / M_IN_K * 10,
                # self.previous_bitrate[index][-1] / 5,
                # self.previous_buff_size[index][-1] / BUFFER_NORM_FACTOR
            ]

        features = np.array([[elem[0], elem[3], elem[1] / 6] for elem in server_state])
        k_means = K_Means(k=self.k)
        k_means.fit(features)

        classf = k_means.clustered_by_type(features, self.device_type)

        vecs = [[] for _ in range(self.k)]
        wss_display = []
        for i in range(self.k):
            for j in range(len(classf[i])):
                vecs[i].append(server_state[classf[i][j]])
        agg_state = []
        for i in range(self.k):
            agg_state.append(np.mean(vecs[i], axis=0))

        wss_current = 0
        for i in range(self.k):
            for j in range(len(classf[i])):
                dist = self.e_dist(list(agg_state[i]), server_state[classf[i][j]])
                wss_current += dist
                wss_display.append(dist)

        server_state = np.array(agg_state)[:, 1:].flatten()

        server_reward, min_agent_index = self.get_latest_reward()
        server_reward = normalize_qoe(server_reward)
        self.reset_previous()
        return server_state, classf, server_reward, min_agent_index, wss_current

    def get_eval_reward(self):
        r_l = []
        for i in range(self.n_agents):
            r_l.append(self.previous_qoe[i][-1])
        return np.min(r_l)

    def e_dist(self, v0: list, v1: list):
        dist = 0
        for i in range(len(v0)):
            dist += (v0[i]-v1[i])**2
        return dist

    def get_wss(self):
        return self.wss
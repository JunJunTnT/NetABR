import os

import numpy as np

from simulator.constants import (
    BITS_IN_BYTE, B_IN_MB, MILLISECONDS_IN_SECOND, TOTAL_VIDEO_CHUNK,
    VIDEO_BIT_RATE)
from simulator.schedulers import Scheduler, TestScheduler
from simulator.utils import construct_bitrate_chunksize_map

DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
RANDOM_SEED = 42


class Environment:
    def __init__(self, trace_scheduler: Scheduler, chunk_len: float,
                 video_size_file_dir: str, random_seed: int = RANDOM_SEED):
        """

            chunk_len: length of a video chunk. Unit: Second
        """
            # all_file_names=None,
                 # random_seed=RANDOM_SEED, link_rtt=80, buff_thres = 60, chunk_len=4, fixed=False):
        # assert len(all_cooked_time) == len(all_cooked_bw)
        # self.traces = traces
        np.random.seed(random_seed)
        self.trace_scheduler = trace_scheduler
        self.chunk_len = chunk_len * MILLISECONDS_IN_SECOND

        self.video_chunk_counter = 0
        self.buffer_size = 0

        self.trace = self.trace_scheduler.get_trace() #s[self.trace_idx]
        # print(np.mean(self.trace))
        # print(np.std(self.trace))
        self.cooked_time = self.trace.timestamps
        self.cooked_bw = self.trace.bandwidths
        # for i in range(len(self.cooked_bw)):
        #     self.cooked_bw[i] = 2

        self.bw_co = 1
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1 \
            # if isinstance(self.trace_scheduler, TestScheduler) else np.random.randint(1, len(self.cooked_bw))
        # randomize the start point of the trace
        # note: trace file starts with time 0
        # print(self.mahimahi_ptr)
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = construct_bitrate_chunksize_map(video_size_file_dir)

        length = len(self.video_size[0])
        for i in range(len(self.video_size)):
            for j in range(length):
                for k in range(19):
                    self.video_size[i].append(self.video_size[i][j])

    def set_bw_co(self, bw_co: float):
        self.bw_co = bw_co


    def get_video_chunk(self, quality: int):

        assert quality >= 0
        assert quality < len(VIDEO_BIT_RATE)

        video_chunk_size = self.video_size[quality][self.video_chunk_counter] / 10
        #print(video_chunk_size, "----video_chunk_size---")

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes
        # print(self.trace.bandwidths)

        while True:  # download video chunk over mahimahi
            # == 信道吞吐量 bit
            throughput = self.cooked_bw[self.mahimahi_ptr] * self.bw_co * B_IN_MB / BITS_IN_BYTE  # throughput = bytes per ms
            #print(self.cooked_bw[self.mahimahi_ptr], "bw")
            # == 计算时间片
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            # 单个时间片可以发送的数据大小
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:        # 单个时间片可发送数据小于chunk大小，则继续添加
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                # print( delay ,"--------fractional_time------" )
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration           # 发送一个chunk所需时间
            #print(delay, "--------fractional_time + duration------")

            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = self.mahimahi_start_ptr
                self.last_mahimahi_time = 0


        #print( delay ,"--------before------" )
        delay *= MILLISECONDS_IN_SECOND     # 发送时延
        #print( delay ,"--------MILLISECONDS_IN_SECOND------" )

        delay += self.trace.link_rtt        # 加上链路时延rtt
        #print( delay ,"--------RTT------" )

        # add a multiplicative noise to the delay
        # if not self.fixed:
        if not isinstance(self.trace_scheduler, TestScheduler):
            delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)
        # print(self.buffer_size)

        # add in the new chunk
        self.buffer_size += self.chunk_len      # 单位为时间 每个chunk对应长度为4s的视频内容

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.trace.buffer_thresh:     # 链路缓冲区溢出时 该链路选择停止传输数据 即sleep
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.trace.buffer_thresh
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME            # 向上取整
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = self.mahimahi_start_ptr
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            self.trace = self.trace_scheduler.get_trace()
            self.cooked_time = self.trace.timestamps
            self.cooked_bw = self.trace.bandwidths

            # for i in range(len(self.cooked_bw)):
            #     self.cooked_bw[i] = 2

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr if not isinstance(self.trace_scheduler, TestScheduler) else np.random.randint(1, len(self.cooked_time))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(len(VIDEO_BIT_RATE)):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain

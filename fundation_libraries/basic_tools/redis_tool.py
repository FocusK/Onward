#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import in_place
import random
import requests
import threading
import tqdm
from retry import retry

from maravilla.infrastructure import constants
from maravilla.infrastructure import shell_tool

class Redis(object):

    def __init__(self, ip, port, schema = "http:/", path = constants.REDIS_BATCH_SERVICE_NAME):
        """
            Args:
                ip     : str, 请求服务所在机器的ip
                port   : str, 服务使用的端口
                schema : str, 使用的协议
                path   : str, 接口名
        """
        self.uri = "/".join([schema, ip + ":" + port, path])
        self.session = requests.Session()
        
    @retry(requests.exceptions.Timeout, tries = 3, delay = 1)
    def set(self, key, value, expired_seconds = 604800, timeout = 3):
        """
            将kv pair写入redis, 同时设置key的过期时间

            Args:
                key             : string, redis中的key
                value           : string, 对应key的value
                expired_seconds : int, key过期时间 
                timeout         : int, 超时时间, 之后会停止等待响应, 避免阻塞
            Returns:
                -
        """
        value_payload = {'cmd': 'SETEX', 'args': [key, str(expired_seconds), value]}
        try:
            response = self.session.post(self.uri, json = value_payload, timeout = timeout)
        except:
            print(f"Redis::set: can not get response after reaching timeout limit {timeout}")
            raise requests.exceptions.Timeout
        if response.status_code != constants.HTTP_SUCCESS_CODE:
            print(f"Redis::set: failed to set the value: {value_payload['args']} to redis, error code: {response.status_code}")
            raise requests.exceptions.HTTPError

    @retry(requests.exceptions.Timeout, tries = 3, delay = 1)
    def get(self, key, timeout = 3):
        """
            在redis中查key对应的value

            Args:
                key     : string, 需要查询的key
                timeout : int, 超时时间, 之后会停止等待响应, 避免阻塞
            Return:
                value   : string, key对应的value
        """
        key_payload = {'cmd': 'GET', 'args': [key]}
        try:
            response = self.session.post(self.uri, json = key_payload, timeout = timeout)
        except:
            print(f"Redis::get: can not get response after reaching timeout limit {timeout}")
            raise requests.exceptions.Timeout
        if response.status_code != constants.HTTP_SUCCESS_CODE:
            print(f"Redis::get: failed to get the value w.r.t. key: {key_payload['args']}, error code: {response.status_code}")
            return ""
        return response.json()

    @retry(requests.exceptions.Timeout, tries = 3, delay = 1)
    def batch_set(self, key_value_list, expired_seconds = 604800, timeout = 3):
        """
            批量将写入kv pair写入redis

            Args:
                key_value_list  : list<list<string>>, 多个kv pair组成的列表
                expired_seconds : int, key过期时间
                timeout         : int, 超时时间, 之后会停止等待响应, 避免阻塞
            Returns:
                -
        """
        payload = {'batch': []}
        for key, value in key_value_list:
            value_payload = {'cmd': 'SETEX', 'args': [key, str(expired_seconds), value]}
            payload['batch'].append(value_payload)
        try:
            response = self.session.post(self.uri, json = payload, timeout = timeout)
        except:
            print(f"Redis::batch_set: can not get response after reaching timeout limit {timeout}")
            raise requests.exceptions.Timeout
        if response.status_code != constants.HTTP_SUCCESS_CODE:
            print(f"Redis::batch_set: failed to set the value: {payload['batch']} to redis, error code: {response.status_code}")
            raise requests.exceptions.HTTPError

    @retry(requests.exceptions.Timeout, tries = 3, delay = 1)
    def batch_get(self, key_list, timeout = 3):
        """
            批量查询redis中key对应的value

            Args:
                key_list : list<string>, 待查询的key列表
                timeout  : int, 超时时间, 之后会停止等待响应, 避免阻塞
            Returns:
                response : str, 返回体
        """
        payload = {'batch': []}
        for key in key_list:
            key_payload = {'cmd': 'GET', 'args': [key]}
            payload['batch'].append(key_payload)
        try:
            response = self.session.post(self.uri, json = payload, timeout = timeout)
        except:
            print(f"Redis::batch_get: can not get response after reaching timeout limit {timeout}")
            raise requests.exceptions.Timeout
        if response.status_code != constants.HTTP_SUCCESS_CODE:
            print(f"Redis::batch_get: failed to get the value w.r.t. key: {payload['batch']}, error code: {response.status_code}")
            return ""
        return response.json()

    def add_prefix(self, filename, prefix, delimiter = "\t"):
        """
            对需要写redis的文件加上key prefix

            Args:
                filename  : str, 待写入的文件名
                prefix    : str, redis key前缀
                delimiter : str, key-value分隔符

            Returns:
                -
        """
        with in_place.InPlace(filename) as f:
            for line in f:
                k, v = line.strip().split(delimiter)
                k = prefix + k
                f.write(delimiter.join([k, v]) + "\n")

    def key_sharding(self, filename, key_list, num_shards, delimiter = "\t", sub_delimiter = ","):
        """
            对指定的存储进行打散, 单个key打散为`num_shards`个key

            Args:
                filename      : str, 文件名
                key_list      : list<str>, 需要打散的key
                num_shards    : int, 打散的份数
                delimiter     : str, kv分隔符
                sub_delimiter : str, value分隔符
            Returns:
                -
        """
        with in_place.InPlace(filename) as f:
            for line in f:
                k, v = line.strip().split(delimiter)
                if k not in key_list:
                    f.write(delimiter.join([k, v]) + "\n")
                else:
                    v_list = v.split(sub_delimiter)
                    num_values = len(v_list)
                    # 划分规则: 首先尽可能均匀填充每个shard, 多余的由前往后填充, 会保证原始顺序
                    num_shards_need_fill = num_values % num_shards
                    for index in range(num_shards):
                        batch_size = num_values // num_shards
                        if index < num_shards_need_fill: 
                            batch_size += 1
                        f.write(delimiter.join([k + f"_{index}", 
                                sub_delimiter.join(v_list[: batch_size])]) + "\n")
                        v_list = v_list[batch_size:]

    def write(self, filename, delimiter = "\t", batch_size = 100, expired_seconds = 604800, shattering_interval = (-300, 300)):
        """
            将文件中的kv pair写入redis

            Args:
                filename        : str, 待写入的文件
                delimiter       : str, key/value分隔符
                batch_size      : int, 如果是批量写入, 每个批次的大小
                expired_seconds : int, key的过期时间
                shattering_interval : tuple, 大量key同时过期会导致redis负载增加, 随机生成区间内随机数+过期时间进行打散

            Returns:
                -
        """ 
        if not os.path.exists(filename):
            print(f"Redis::write: can not find file {filename}")
            raise FileNotFoundError
        total_num_lines = int(shell_tool.execute_with_response(f"wc -l {filename}").split(" ")[0])
        with open(filename, encoding = "utf8") as f:
            key_value_list = []
            for line in tqdm.tqdm(f, total = total_num_lines):
                try:
                    key, value = line.strip().split(delimiter)
                except:
                    continue
                key_value_list.append([key, value])
                if len(key_value_list) >= batch_size:
                    self.batch_set(key_value_list, expired_seconds + random.randint(*shattering_interval))
                    key_value_list = []
            if key_value_list:
                self.batch_set(key_value_list, expired_seconds + random.randint(*shattering_interval))

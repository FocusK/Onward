#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
from ctypes import cdll, c_char_p

from maravilla.infrastructure import constants

class BNSResolver(object):
    """
        提供域名到ipport的映射
    """
    def __init__(self, bns_library_path = constants.BNS_LIBRARY_PATH):
        """
            初始化实例

            Args:
                bns_library_path : str, 解析bns的so文件路径
            Returns:
                - 
        """
        self.bns_library = cdll.LoadLibrary(bns_library_path)

    def get_machine_list_from_bns(self, bns):
        """
            根据服务名获取运行该服务的机器列表

            Args:
                bns : str, bns名称

            Returns:
                machine_dict_list : list<dict>, 查询到该bns下的机器列表
        """
        self.bns_library.getBNS.argtype = c_char_p
        self.bns_library.getBNS.restype = c_char_p
        response_dict = eval(self.bns_library.getBNS(bns.encode("utf-8")).decode("utf-8"))
        try:
            errno = response_dict["errno"]
        except:
            print(f"BNSResolver::get_machine_list_from_bns: response does not contain `errno`")
            raise KeyError
        if errno != 0:
            print(f"BNSResolver::get_machine_list_from_bns: wrong response status, errno = {errno}")
            raise ValueError
        try:
            machine_dict_list = response_dict["data"]
        except:
            print(f"BNSResolver::get_machine_list_from_bns: response does not contain `data`")
            raise KeyError
        if len(machine_dict_list) == 0:
            print(f"BNSResolver::get_machine_list_from_bns: no available machine under the bns {bns}")
            raise ValueError
        return machine_dict_list
    
    def get_random_ipport(self, bns):
        """
            在给定BNS下所有机器中随机选择一台, 返回ip, port

            Args:
                bns : str, bns名称
            Return:
                ip   : str, 返回机器的ip
                port : str, 服务的port
        """
        machine_dict_list = self.get_machine_list_from_bns(bns)
        machine_dict = random.choice(machine_dict_list)
        ip, port = machine_dict["Host"], machine_dict["Port"]
        return ip, str(port)

    def choose_random_ipport(self, machine_dict_list):
        """
            从给定机器列表中随机选择一台, 返回ip, port

            Args:
                machine_list: list<dict>, 机器列表
            Return:
                ip   : str, 机器的ip
                port : str, 服务的port
        """
        machine_dict = random.choice(machine_dict_list)
        ip, port = machine_dict["Host"], machine_dict["Port"]
        return ip, str(port)

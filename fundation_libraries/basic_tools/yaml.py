#!/usr/bin/python
# -*- coding: utf-8 -*-

import os

import yaml

def load_yaml(yaml_path):
    """
        加载yaml文件获取配置字典
        Args: 
            yaml_path: str, yaml文件所在路径
        Returns:
            resolved_dict: dict, 解析yaml后得到的字典
    """
    if not os.path.exists(yaml_path):
        print(f"YAML::load_yaml: the path {yaml_path} does not exist")
        raise FileNotFoundError
    try:
        with open(yaml_path, encoding='utf-8') as f:
            resolved_dict = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print(f"YAML::load_yaml: failed to resolve {yaml_path}, please check the format of YAML.")
        raise IOError
    if resolved_dict == None:
        print(f"YAML::load_yaml: the content of {yaml_path} is None, please ensure the content is correct")
        raise EOFError
    return resolved_dict


#!/usr/bin/python
# -*- coding: utf-8 -*-
from maravilla.infrastructure import constants
from maravilla.infrastructure import shell_tool

class BinaryPbdictCreator(object):

    def __init__(self, 
        pbdict_library_path = constants.PBDICT_LIBRARY_PATH,
        proto_compiler_bin = constants.PROTO_COMPILER_PATH):

        """
            初始化二进制PB词典生成器

            Args:
                pbdict_library_path : str, 二进制词典生成器可执行文件的路径
                proto_compiler_bin  : str, protobuf编译器路径
            Returns:
                -
        """
        self.pbdict_library_path = pbdict_library_path
        self.proto_compiler_bin = proto_compiler_bin

    def create_binary_pbdict(self, 
        pbdict_path,
        pbdict_name,
        binary_pbdict_path,
        binary_pbdict_name,
        proto_file, 
        max_num,
        message,
        delimiter = "\\\t"):

        """
            创建二进制的ProtoBuffer词典

            Args:
                pbdict_path        : str, 原始词典路径
                pbdict_name        : str, 原始词典文件名
                binary_pbdict_path : str, 二进制词典路径
                binary_pbdict_name : str, 二进制词典文件名
                proto_file         : str, pbdict的proto格式文件名
                max_num            : int, 原始词典允许的最大行数
                message            : str, 词典信息
                delimiter          : str, 分隔符

            Returns:
                -
        """
        command_dict = {
            "-p": pbdict_path,
            "-n": pbdict_name,
            "-P": binary_pbdict_path,
            "-N": binary_pbdict_name,
            "-B": proto_file,
            "-m": str(max_num),
            "-b": message,
            "-s": delimiter}
        command_list = []
        for key in command_dict.keys():
            command_list.append(key)
            try:
                command_list.append(command_dict[key])
            except:
                print(f"BinaryPbdictCreator::create_binary_pbdict: the command_dict does not has key {key}")
                raise KeyError
        command_str = self.pbdict_library_path + " " + " ".join(command_list)
        if shell_tool.execute_with_errorcode(command_str):
            print(f"BinaryPbdictCreator::create_binary_pbdict: failed to execute the command {command_str}")
            raise ValueError

    def compile_proto_file(self, proto_path, proto_file, output_path):
        """
            编译protobuf文件

            Args:
                proto_path  : str, proto文件目录
                proto_file  : str, proto文件名  
                output_path : str, 编译后proto输出目录
        """ 
        command_str = f"{self.proto_compiler_bin} -I={proto_path} --python_out=\"{output_path}\" {proto_file}"
        if shell_tool.execute_with_errorcode(command_str):
            print(f"BinaryPbdictCreator::compile_proto_file: failed to execute the command {command_str}")
            raise ValueError

#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import pathlib

from maravilla.infrastructure import shell_tool
from maravilla.infrastructure import time_tool

class FileSystem(object):

    def __init__(self, 
        root,
        user_home = "/home/users/chenxingke"):
        
        """
            初始化文件系统工具的配置

            Args:
                root      : str, 文件系统工具能够操作的根目录
                user_home : str, 用户的家目录, 原则上用户只能在自己的家目录下操作
        """
        
        self.root = pathlib.Path(root)
        self.user_home = pathlib.Path(user_home)

    def get_absolute_path(self, path):
        """
            获取给定路径的绝对路径

            Args:
                path          : str, 任意posix路径
                absolute_path : pathlib.Path, 对应绝对路径
        """
        # 如果是绝对路径, 保证给定的目录在user_home且root中
        if path.startswith("/"):
            if str(self.user_home) not in path:
                print(f"FileSystem::get_absolute_path: {path} may not in your own directory")
                raise PermissionError
            elif str(self.root) not in path:
                print(f"FileSystem::get_absolute_path: {path} is not in the root directory {self.root}")
                raise PermissionError
            return pathlib.Path(path)
        # 如果是相对路径, 保证root目录在user_home中, 以root为基准进行扩展
        else:
            if str(self.user_home) not in str(self.root):
                print(f"FileSystem::get_absolute_path: the root directory {self.root} is not in your home {self.user_home}")
                raise PermissionError
            # 相对路径, 基准为root
            return self.root / path
    
    def operation_path(self, path):
        path = self.get_absolute_path(path)
        return str(path)

    def operation_relpath(self, path, base_path):
        """
            获取某个路径相对基准路径的相对路径
        """
        path = self.get_absolute_path(path)
        base_path = self.get_absolute_path(base_path)
        return os.path.relpath(base_path, path)

    def operation_exist(self, path):
        path = self.get_absolute_path(path)
        if path.exists():
            return True
        return False
 
    def operation_mkdir(self, path, is_file = False, mode = 0o777, exist_ok = True, parents = True):
        """
            创建目录, 如果传入文件, 默认创建父目录

            Args:
                path     : str, 待创建的目录路径
                is_file  : bool, 路径是否为文件
                mode     : int, 访问权限
                exist_ok : bool, 当目标目录存在时
                
        """
        path = self.get_absolute_path(path)
        if is_file:
            path = path.parent
        try:
            path.mkdir(mode = mode, exist_ok = exist_ok, parents = parents)
        except:
            print(f"FileSystem::operation_mkdir: failed to create the directory {path}")
            return 1
        return 0

    def operation_rm(self, path):
        """
            删除指定路径(文件、链接或目录)
        """
        path = self.get_absolute_path(path)
        if not path.exists():
            print(f"FileSystem::operation_rm: path {path} does not exist")
            return
        # 对于文件或链接, 直接删除
        if path.is_file() or path.is_symlink():
            path.unlink()
        # 对于目录, 执行递归删除
        elif path.is_dir():
            self.util_rmtree(path)
        else:
            print(f"FileSystem::operation_rm: unknown object type")
            raise ValueError
        print(f"FileSystem::operation_rm: succeeded to delete the path {path}")

    def operation_stat(self, path): 
        """
            获取文件信息

            Args:
                path : str, 文件或目录的路径
            Returns:
                stat_dict : dict, 字典形式的文件信息
        """
        path = self.get_absolute_path(path)
        if not path.exists():
            print(f"FileSystem::operation_stat: path {path} does not exist")
            raise ValueError
        stat = path.stat()
        stat_dict = {info: getattr(stat, info) for info in dir(stat) if info.startswith('st_')}
        return stat_dict

    def operation_mtime(self, path):
        """
            获取文件最近一次修改的时间

            Args:
                path  : str, 文件或目录的路径
            Returns:
                mtime : str, 修改时间的字符串形式
        """
        stat_dict = self.operation_stat(path)
        mtime_timestamp = int(stat_dict["st_mtime"])
        mtime = time_tool.timestamp_to_string(mtime_timestamp)
        return mtime

    def operation_size(self, path):
        path = self.get_absolute_path(path)
        if path.is_file() or path.is_symlink():
            stat_dict = self.operation_stat(str(path))
            size = stat_dict["st_size"]
            return size
        size = 0 
        for child_path in self.util_walk(path):
            size += self.operation_stat(str(child_path))["st_size"]
        return size

    def operation_tar(self, source, target, optional = "-czvf"):
        """
            对目标文件进行打包压缩
            
            Args:
                source   : str, 需要打包目录的路径
                target   : str, 打包文件的产出路径
                optional : str, 打包的可选参数
                       z 表示压缩
                       c 表示打包
                       x 表示解压
                       v 表示显示详情
                       f 表示对应文件名
            Returns:
                -
        """
        source = self.get_absolute_path(source)
        target = self.get_absolute_path(target)
        command_str = f"tar {optional} {target} -C {source.parent} {source.name}"
        if shell_tool.execute_with_errorcode(command_str):
            print(f"FileSystem::operation_tar: failed to tar the object {source}")
            raise ValueError

    def operation_md5sum(self, source, target):
        """
            生成文件的md5码 

            Args:
                source : str, 原始文件路径
                target : str, 生成md5码的路径

            Returns:
                -
        """
        source = self.get_absolute_path(source)
        target = self.get_absolute_path(target)
        command_str = f"md5sum {source} | awk -F/  \'{{print $1  $NF}}\' > {target}"
        if shell_tool.execute_with_errorcode(command_str):
            print(f"FileSystem::operation_md5sum: failed to generate md5sum for {source}")
            raise ValueError

    def operation_filename(self, path):
        """
            获取给定路径的文件名
        """
        path = self.get_absolute_path(path)
        return str(path.name)

    def operation_parent(self, path, level = 0):
        """
            获取给定路径的父级目录

            Args:
                path  : str, 目标路径
                level : int, 目录层级, 0表示父节点
            Returns:
                parent_directory : str, 父目录     
        """
        path = self.get_absolute_path(path)
        try:
            parent_directory = path.parents[level]
        except:
            print(f"FileSystem::operation_parent: the path {path} does not have level {level} parent")
            raise ValueError
        return str(parent_directory)

    def operation_stem(self, path):
        path = self.get_absolute_path(path)
        return path.stem

    def operation_file_lines(self, path):
        path = self.get_absolute_path(path)
        if not path.exists():
            print(f"FileSystem::operation_file_lines: path {path} does not exist")
            raise ValueError
        if not path.is_file():
            print(f"FileSystem::operation_file_lines: path {path} is not a file")
            raise TypeError
        command_str = f"wc -l {path}"
        response_str = shell_tool.execute_with_response(command_str).split(" ")[0]
        num_lines = int(response_str)
        return num_lines

    def util_rmtree(self, path):
        """
            删除整个目录

            Args:
                path : pathlib.Path, 待删除的目录
            Returns:
                -
        """
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            for child_path in path.iterdir():
                self.util_rmtree(child_path)
            path.rmdir()

    def util_walk(self, path):
        for child_path in path.iterdir():
            if child_path.is_dir():
                yield from self.util_walk(child_path)
                continue
            yield child_path.resolve()

#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

from maravilla.infrastructure import bns_tool
from maravilla.infrastructure import constants
from maravilla.infrastructure import time_tool
from maravilla.infrastructure import fs_tool
from maravilla.infrastructure import hdfs_tool
from maravilla.infrastructure import redis_tool

# const
basetime_bias = 0
basetime = time_tool.get_date_before_today(basetime_bias)
user2tag_redis_prefix = "dueros:edu_pad:sv:user2tag:"
tag2item_redis_prefix = "dueros:edu_pad:sv:tag2item:"
expired_days = 3

# 本地目录
user_home = constants.USER_HOME
root = f"{user_home}/edupad/recall/content_based_recall/{basetime}/"
user2tag_file = "user2tag.txt"
tag2item_file = "tag2item.txt"

# HDFS目录
user2tag_directory = f"edupad/recall/content_based/1_edupad_user2tag/{basetime}/"
tag2item_directory = f"edupad/recall/content_based/2_edupad_tag2item/{basetime}/"
user2tag_backup_directory = f"edupad/recall/content_based/1_edupad_user2tag/backup/{basetime}/"
tag2item_backup_directory = f"edupad/recall/content_based/2_edupad_tag2item/backup/{basetime}/"

def main():
    hdfs = hdfs_tool.HDFS(queryengine_bin = constants.QUERYENGINE_PATH, basetime_bias = basetime_bias)
    file_system = fs_tool.FileSystem(root, user_home)
    bns = bns_tool.BNSResolver()
    ip, port = bns.get_random_ipport(constants.REDIS_BNS)
    redis = redis_tool.Redis(ip, port)
    if file_system.operation_exist(root):
        file_system.operation_rm(root)
    file_system.operation_mkdir(root) 
    # 获取用户画像灌库
    file_system.operation_mkdir(user2tag_file, is_file = True)
    if hdfs.operation_queryengine("./1_edupad_user2tag.yml"):
        print("0_main_procedure: failed to execute 2_edupad_tag2item job")
        sys.exit(1)
    hdfs.operation_get(user2tag_directory, file_system.operation_path(user2tag_file), True)
    file_system.operation_rm("." + user2tag_file + ".crc")
    redis.add_prefix(file_system.operation_path(user2tag_file), user2tag_redis_prefix)
    redis.write(file_system.operation_path(user2tag_file))
    if hdfs.operation_put(
            user2tag_backup_directory,
            file_system.operation_path(user2tag_file)):
        print(f"0_main_procedure: failed to put {file_system.operation_parent(user2tag_file)} to backup directory")

    # 获取每个tag下对应的资源灌库
    if hdfs.operation_queryengine("./2_edupad_tag2item.yml"):
        print("0_main_procedure: failed to execute 3_edupad_tag2item job")
        sys.exit(1) 
    hdfs.operation_get(tag2item_directory, file_system.operation_path(tag2item_file), True) 
    file_system.operation_rm("." + tag2item_file + ".crc")
    redis.add_prefix(file_system.operation_path(tag2item_file), tag2item_redis_prefix)
    redis.write(file_system.operation_path(tag2item_file))
    if hdfs.operation_put(
            tag2item_backup_directory,
            file_system.operation_path(tag2item_file)):
        print(f"0_main_procedure: failed to put {file_system.operation_parent(tag2item_file)} to backup directory")
    # 清理环境
    for hdfs_directory in [user2tag_directory, tag2item_directory]:
        hdfs.operation_delete(hdfs_directory)
    for hdfs_directory in [user2tag_backup_directory, tag2item_backup_directory]:
        hdfs.operation_clean(hdfs.operation_parent_path(hdfs_directory), expired_days)
    file_system.operation_rm(root)
    print("0_main_procedure: succeeded to execute all jobs")
    sys.exit(0)

if __name__ == "__main__":
    main()

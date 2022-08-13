#!/usr/bin/python
# -*- coding: utf-8 -*-

# time
DAY_SECONDS = 86400
DAY_HOURS = 24

# network
HTTP_SUCCESS_CODE = 200

# bns
RUS_BNS = "group.dumi-DuRus.dumi.all"
GRS_BNS = "group.dumi-GlobalRs.dumi.all"
FS_BNS = "group.dumi-FeatureService.dumi.all"
RS_SHORT_VIDEO_BNS = "group.dumi-RsShortVideo.dumi.all"

ES_VOD_BNS = "smartbns#bjyz@group.duer-es-botcommon-sandbox.dumi.nj"
ES_SV_BNS = "smartbns#bjyz@group.duer-es-botcommon-sandbox.dumi.nj"
ES_MUSIC_BNS = "group.duer-es-music.dumi.all"

REDIS_BNS = "group.dumi-bdrp-proxy.dumi.all"
REDIS_SANDBOX_BNS = "group.dumi-bdrp-proxy-sandbox.dumi.all"

# service name
RUS_SERVICE_NAME = "rus"
FS_SERVICE_NAME = "GetFeatureService/get_homefeed_feature"
## ES各垂类服务
ES_VOD_SERVICE_NAME = "duer_video_on_demand/video_on_demand/_search"
ES_SV_SERVICE_NAME = "duer_video/video_track/_search"
ES_MUSIC_SERVICE_NAME = "music_song_full/music_song/_search"
## ES获取视频OST服务
ES_OST_SERVICE_NAME = "music_song_full/_search"
## redis
REDIS_BATCH_SERVICE_NAME = "DproxyServer/batch"
REDIS_CMD_SERVICE_NAME = "DproxyServer/cmd"

# request path
REQUEST_PATH_PREFIX = "/home/users/chenxingke/maravilla/data/request_templates/" 
RUS_REQUEST_PATH = REQUEST_PATH_PREFIX + "rus_request.txt"
FS_REQUEST_PATH = REQUEST_PATH_PREFIX + "fs_request.txt"
ES_SV_REQUEST_PATH = REQUEST_PATH_PREFIX + "es_short_video_request.txt"
ES_VOD_REQUEST_PATH = REQUEST_PATH_PREFIX + "es_video_on_demand_request.txt"
ES_MU_REQUEST_PATH = REQUEST_PATH_PREFIX + "es_music_request.txt"
ES_OST_REQUEST_PATH = REQUEST_PATH_PREFIX + "es_ost_request.txt"

# udf path
UDF_PATH_PREFIX = "/home/users/chenxingke/maravilla/toolbox/udf/"
TIME_UDF_PATH = UDF_PATH_PREFIX + "time_udf.py"
SORT_UDF_PATH = UDF_PATH_PREFIX + "sort_udf.py"

JAR_UDF_PATH = "afs://pegasus.afs.baidu.com:9902/user/dumi_data_platform/lib/udf-1.0-SNAPSHOT.jar"

# proto path
PROTO_PATH_PREFIX = "/home/users/chenxingke/maravilla/data/protos/"
## homefeed
USER2USER_PROTO_PATH = PROTO_PATH_PREFIX + "homefeed/user2user_recom.proto"
LONG_HISTORY_PROTO_PATH = PROTO_PATH_PREFIX + "homefeed/user_long_term_consume_history.proto"

# bin path
BIN_PATH_PREFIX = "/home/users/chenxingke/maravilla/infrastructure/bin/"
BNS_LIBRARY_PATH = BIN_PATH_PREFIX + "bns/get_bns.so"
DICTSET_LIBRARY_PATH = BIN_PATH_PREFIX + "dictset/create_binary_dset"
PBDICT_LIBRARY_PATH = BIN_PATH_PREFIX + "pbdict/create_binary_pbdict"

QUERYENGINE_PATH = "/home/work/env/QueryEngine/queryengine-client-2.1.27-online/queryengine/bin/queryengine --sessionconf engine=wing"
HADOOP_PATH = "/home/users/chenxingke/tools/hadoop/output/hadoop-client/hadoop/bin/hadoop"
PROTO_COMPILER_PATH = "/home/work/.jumbo/bin/protoc"

# data path
USER_HOME = "/ssd1/work/chenxingke/data"

# rus
HOMEFEED_DISTRIBUTE_NUMBER = 54

BOT_ID_SET = {"ai.dueros.bot.video_on_demand", 
              "ai.dueros.bot.short_video", 
              "audio_news", 
              "audio_music", 
              "audio_unicast"}

# fs
HOMEFEED_FEATURE_SET = {"short_term_play_hist_feature",
                        "user_long_term_consume_history_feature",
                        "user_prefer_tag_feature",
                        "user_poi_tag_feature",
                        "similarity_user_long_term_consume_history_feature",
                        "user_base_feature",
                        "feed_showlist_feature",
                        "muses_endless_showlist_feature",
                        "muses_endless_pagelist_feature",
                        "user_prediction_tag_feature"}



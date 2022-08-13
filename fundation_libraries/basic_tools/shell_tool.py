#!/usr/bin/python
# -*- coding: utf-8 -*-

import subprocess

def execute_with_response(command_str):
    res = subprocess.getoutput(command_str)
    return res

def execute_with_errorcode(command_str):
    res = subprocess.call(command_str, shell = True)
    return res

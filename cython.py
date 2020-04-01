#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : build.py
# @Author: Wade Cheung
# @Date  : 2019/2/23
# @Desc  : 使用Cython.Build.cythonize将py编译成.so文件


import sys
import os
import shutil
from distutils.core import setup
from Cython.Build import cythonize

currdir = os.path.abspath('.')
parentpath = sys.argv[1] if len(sys.argv) > 1 else ""
setupfile = os.path.join(os.path.abspath('.'), __file__)
build_dir = "build"
build_tmp_dir = build_dir + "/temp"


def getpy(basepath=os.path.abspath('.'), parentpath='', name='', excepts=(),
          copyOther=False, delC=False):
    """
    获取py文件的路径
    :param basepath: 根路径
    :param parentpath: 父路径
    :param name: 文件/夹
    :param excepts: 排除文件
    :param copy: 是否copy其他文件
    :return: py文件的迭代器
    """
    fullpath = os.path.join(basepath, parentpath, name)
    for fname in os.listdir(fullpath):
        ffile = os.path.join(fullpath, fname)
        # print(basepath, parentpath, name,file)
        if os.path.isdir(ffile) and fname != build_dir and not fname.startswith(
                '.'):
            for f in getpy(basepath, os.path.join(parentpath, name), fname,
                           excepts, copyOther, delC):
                yield f
        elif os.path.isfile(ffile):
            ext = os.path.splitext(fname)[1]
            # 删除.c 临时文件
            if ext == ".c":
                if delC:
                    os.remove(ffile)
            elif ffile not in excepts and os.path.splitext(fname)[1] not in (
                    '.pyc', '.pyx'):
                flag1 = os.path.splitext(fname)[1] in ('.py', '.pyx')
                if flag1:  # and not fname.startswith('__')
                    yield os.path.join(parentpath, name, fname)
                elif copyOther:  # 复制其他文件到./build 目录下
                    dstdir = os.path.join(basepath, build_dir, parentpath, name)
                    if not os.path.isdir(dstdir):
                        os.makedirs(dstdir)
                    shutil.copyfile(ffile, os.path.join(dstdir, fname))
        else:
            pass


# 获取py列表
module_list = list(
    getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile,)))
print(module_list)

# 编译成.so文件
try:
    setup(ext_modules=cythonize(module_list),
          script_args=["build_ext", "-b", build_dir, "-t", build_tmp_dir])
except Exception as ex:
    print("error! ", str(ex))
else:
    # 复制其他文件到./build 目录下
    module_list = list(
        getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile,),
              copyOther=True))

# 删除临时文件 ~
module_list2 = list(
    getpy(basepath=currdir, parentpath=parentpath, excepts=(setupfile,),
          delC=True))
if os.path.exists(build_tmp_dir):
    shutil.rmtree(build_tmp_dir)

print("Done ! ")
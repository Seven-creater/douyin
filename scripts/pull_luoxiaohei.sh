#!/bin/bash
# 罗小黑素材：ModelScope 拉取 + MD5 记录 + 解压一条龙（服务器侧，参数1=签名URL）
# 用法: setsid nohup ./pull_luoxiaohei.sh '<URL>' > pull_lxh.log 2>&1 < /dev/null &
set -x
cd /data02/usr/wangqihao/Demo/research/data/raw || exit 1

aria2c -x8 -s8 -c -o luoxiaohei.zip "$1" >> dl_lxh.log 2>&1 \
  || { echo ARIA2_FAILED >> dl_lxh.log; exit 1; }

echo MD5_START >> dl_lxh.log
md5sum luoxiaohei.zip >> dl_lxh.log 2>&1

echo UNZIP_START >> dl_lxh.log
if command -v unzip >/dev/null 2>&1; then
  unzip -o -q luoxiaohei.zip -d /data02/usr/wangqihao/Demo/research/data/raw >> dl_lxh.log 2>&1
else
  python3 -m zipfile -e luoxiaohei.zip /data02/usr/wangqihao/Demo/research/data/raw >> dl_lxh.log 2>&1
fi

echo LISTING >> dl_lxh.log
ls -lh /data02/usr/wangqihao/Demo/research/data/raw/luoxiaohei/ >> dl_lxh.log 2>&1
echo ALL_DONE >> dl_lxh.log

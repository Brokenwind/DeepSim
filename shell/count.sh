#!/bin/bash

echo "蚂蚁金服数据情况"
pos=`cat ../data/ant.csv | awk -F '\t' '$4==1{print $0}' | wc -l`
echo "正例: "$pos
neg=`cat ../data/ant.csv | awk -F '\t' '$4==0{print $0}' | wc -l`
echo "反例: "$neg

echo -e "\n微众银行数据情况"
pos=`cat ../data/weichat.csv | awk -F '\t' '$4==1{print $0}' | wc -l`
echo "正例: "$pos
neg=`cat ../data/weichat.csv | awk -F '\t' '$4==0{print $0}' | wc -l`
echo "反例: "$neg

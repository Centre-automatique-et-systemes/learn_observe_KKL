#!/bin/bash
for ((i=0; i<=40; i++))
do
  python experiments/4_supervised_qqs2_meas12.py $i
done
#!/bin/bash
for ((i=32; i<=40; i++))
do
  python experiments/4_supervised_qqs2_meas12.py $i
done
python experiments_noise/4_supervised_qqs2_meas12.py
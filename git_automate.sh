#!/bin/bash

cd /home/iiser/Saptarshi/nerveFlow

git add .

git commit -m $(printf "%s" $(date +"%h_%d_%y_%r"))

git push origin master

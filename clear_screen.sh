#!/bin/bash

screen -ls | grep Detached | awk '{print $1}' | xargs -r -I {} screen -S {} -X quit
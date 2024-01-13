#!/bin/bash

pid=$1  # Process ID passed as an argument
max_mem=0

while true; do
    mem_usage=$(ps -o rss= -p $pid | grep -v RSS)  # Get RSS memory usage of the process, excluding the header

    if ! [[ "$mem_usage" =~ ^[0-9]+$ ]]; then
        echo "Process $pid not found or terminated."
        exit 1
    fi

    if [ "$mem_usage" -gt "$max_mem" ]; then
        max_mem=$mem_usage
    fi

    echo "Current memory usage: $mem_usage kB, Maximum memory usage: $max_mem kB"
    sleep 60
done

#!/bin/bash

if [ -z "$1" ]; then
    # No PID provided, find "pirated" by process name
    pids=$(pgrep -f "pirated")

    if [ -z "$pids" ]; then
        # No 'pirated' process found, ask for PID
        read -p "Enter the PID of the 'pirated' process: " pids
        if [ -z "$pids" ]; then
            echo "No PID entered. Exiting."
            exit 1
        fi
    else
        echo "Found 'pirated' with PIDs: $pids"
    fi
else
    pids=$1  # use PID passed as an argument, if it exists
fi

echo "Starting 60s Memory Monitor Loop for 'pirated'"
max_mem=0

while true; do
    total_mem_usage=0

    for pid in $pids; do
        mem_usage=$(ps -o rss= -p $pid | grep -v RSS | awk '{print $1}')  # Get RSS memory usage of the process, excluding the header
        if ! [[ "$mem_usage" =~ ^[0-9]+$ ]]; then
            echo "Process $pid not found or terminated."
            continue
        fi
        total_mem_usage=$((total_mem_usage + mem_usage))
    done

    total_mem_usage_gb=$(echo "scale=2; $total_mem_usage / 1024 / 1024" | bc)  # Convert to GB
    if [ $(echo "$total_mem_usage > $max_mem" | bc) -eq 1 ]; then
        max_mem=$total_mem_usage
        max_mem_gb=$(echo "scale=2; $max_mem / 1024 / 1024" | bc)  # Convert to GB
    fi

    echo "Current: $total_mem_usage_gb GB | Peak: $max_mem_gb GB"
    sleep 60
done

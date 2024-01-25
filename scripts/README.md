## Scripts

### monitor.sh 
Monitor RAM utilization of pirate-qt or pirated.

Start pirated (or pirate-qt), then start the script with 

```
./monitor.sh
```

It will automatically attempt to detect the process named 'pirated'. If found, the script will print current and peak RSS every minute 

---

Alternatively, you can manually pass the PID to monitor as an argument. 

example, if your PID was 81962:

```BASH
./monitor.sh 81926
```

---



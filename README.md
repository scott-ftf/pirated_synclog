!['pirated_synLogger](https://i.imgur.com/Qwdrn0K.png)

# Pirate Daemon Sync Logger

A simple python3 script for logging and visualizing the network sync process for the Pirate Chain daemon. 

This script runs in the background while the daemon syncs with the network. It collects telemetry, provides a live output, then returns reports and exits when the daemon becomes in sync with the network. Upload the data.csv for an interactive chart of the resources utilized while sync'ing. 

This has not been tested in an environment other than Ubuntu LTS

---

## Get Started 

- Install/build the Pirate Chain daemon (or stop if already running)
- Delete the blocksfolder, chainstate folder, and peers.dat 

Download the script
```bash
# create a directory to work from
mkdir ~/pirated_sync
cd ~/pirated_sync

# Fetch the script
wget https://raw.githubusercontent.com/scott-ftf/pirated_synclog/main/pirated_synclog.py
```
Make executable for your user
```bash
sudo chmod u+x pirated_synclog.py
```

install dependencies
```bash
pip install pandas matplotlib
```

Edit the configuration section in the first few lines of code to define CLI location, datadir location, etc

---

## **TERMINAL SESSION 1** - Start the Sync Logger Script

Run this script in the background
```bash
nohup python3 ~/pirated_sync/pirated_synclog.py start >/dev/null 2>&1 &
```

Monitor a live feed of the sync progress by watching the output log
```bash
tail -f ~/pirated_sync/sync.log
```

The script will automatically generate reports and exit once the Pirate daemon is in sync with the network

To forcefully stop the script, type:
```bash
python3 pirated_synclog.py stop
```

---

## **TERMINAL SESSION 2** - Start the Pirate daemon 

After starting the sync logger, it will listen for the Pirate daemon to start. 

#### *NOTE: this script requires the daemon output in the debug.log. setting `printtoconsole` will prevent the deamon output from being directed to the debug.log.*   

Start the Pirate daemon disowned in the background with nohup:
```bash
nohup ~/pirate/src/pirated >/dev/null 2>&1 &
```
`1>/dev/null 2>/dev/null` redirects both the standard output (1>) and standard error (2>) to /dev/null, effectively discarding them. (suppress pirated's output). 

The daemons output can still be monitoried by watching the log
```bash
tail -f ~/.komodo/PIRATE/debug.log
```

---

## Reports and Logs

Each time the script is initialized, it will create a new folder in the output directory, named with the date and a timestamp.

Once the daemon is in sync with the network, the script will generate a text summay of the sync, a csv of the data samples, a log of any errors, and create a chart.png in the output directory before exiting quietly.

upload the CSV here for a interactive chart of the logged data:

## [https://scott-ftf.github.io/pirated_synclog/](https://scott-ftf.github.io/pirated_synclog/)

<br />
OPTIONAL: Run the chart HTML locally


```bash
# fetch the HTML
wget https://raw.githubusercontent.com/scott-ftf/pirated_synclog/main/index.html

# serve locally
python3 -m http.server --bind 127.0.0.1 7777
```

---

<img src="https://raw.githubusercontent.com/PirateNetwork/mediakit/main/Wordmark/SVG/Pirate_Logo_Wordmark_Gold.svg" style="width:150px;margin:40px auto;display:block;">
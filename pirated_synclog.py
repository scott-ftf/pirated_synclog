import os
import sys
import signal
import csv
import time
import subprocess
import json
import logging
import platform
from datetime import datetime, timedelta, timezone
import psutil
import shutil
import queue
import traceback
from threading import Thread, Event, active_count

# Check for pandas and matplotlib (all others are in the standard library)
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("The pandas and matplotlib modules are required.\nInstall them with 'sudo apt install python3-pandas python3-matplotlib'")
    exit()

### USER CONFIGURATION ###
home = os.path.expanduser("~")
datadir = os.path.join(home, '.komodo/PIRATE') # location of the daemon datadir (typically /home/$USER/.komodo/PIRATE)
CLI =  os.path.join(home, 'pirate/pirate-cli') # location of pirate_cli
sample_rate = 1 # how many minutes between data collection loops
debug_mode = True # logs more messages to the debug.log

# prepare some flags
startup_data = {
    'rescanning': False,
    'starting': False,
    'downloading_bootstrap': False,
    'extracting_blocks': False,
    'building_witness': False,
    'bootstrap_used': False,
    'PIRATEversion': False
} 

# Define a filter that allows messages from DEBUG up to a maximum level
class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        self.max_level = max_level

    def filter(self, record):
        # max_level log message not included in the log file of lower level messages
        return record.levelno < self.max_level

# Create a logger
logger = logging.getLogger('output_logger')

# configure the logger to handle debug.log, error.log, and sync.log (info)
def configureLogging():

    # Create a formatter for debug and error handlers
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')

    # Create a file handler for writing debug messages
    debug_handler = logging.FileHandler(debug_file)
    debug_handler.setLevel(logging.DEBUG)

    # Add filter to the debug_handler
    debug_handler.addFilter(MaxLevelFilter(logging.INFO))
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    # Create a file handler for writing error messages
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    # Create a file handler for writing info messages
    simple_formatter = logging.Formatter('%(message)s')
    info_handler = logging.FileHandler(output_file)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(MaxLevelFilter(logging.ERROR))  # Only allow up to INFO level
    info_handler.setFormatter(simple_formatter)
    logger.addHandler(info_handler)

    # Set the logger level to the lowest level handler (DEBUG in this case)
    logger.setLevel(logging.DEBUG)

# write extra messages to the debug file when debug_mode = True
def debug(*args):
    if debug_mode:
        message = ' '.join(map(str, args))
        logger.debug(message)

# Normal message
def msg(*args):
    message = ' '.join(map(str, args))
    logger.info(message)

# Error message
def err(*args):
    message = ' '.join(map(str, args))
    message += "\nTraceback:\n"
    message += traceback.format_exc()  # Get traceback information
    logger.error(message)

# Event flag to signal the startup_worker to exit and data collection to start
startup_complete = Event()  
daemon_detected = Event()  

# sleep for the remainder of the loop wait time
def sleep_for_interval(start_time):
    interval = sample_rate * 60
    # Time since the initial start
    elapsed_time = time.time() - start_time

    # get the remaining time until the next interval
    sleep_time = interval - (elapsed_time % interval)
    debug(f"Sleeping for {sleep_time} seconds until next sample")

    # Wait for the remaining time until the next interval
    time.sleep(sleep_time)

# simple function for formatting minutes to days, hours, minutes
from datetime import timedelta

# simple function for formatting minutes to days, hours, minutes
def minutes_to_readable_time(total_minutes):
    # Convert total time in minutes to timedelta
    total_time_td = timedelta(minutes=int(total_minutes))

    days = total_time_td.days
    hours, remainder = divmod(total_time_td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60) 

    # Only display units of time that are relevant
    time_string = ""
    if days > 0:
        # add 's' if days > 1 for plural form
        time_string += f"{days} day{'s' if days > 1 else ''} "
    if hours > 0 or days > 0:  # Show hours if there are any, or if there are days
        # add 's' if hours > 1 for plural form
        time_string += f"{hours} hour{'s' if hours > 1 else ''} "
    # add 's' if minutes > 1 for plural form
    time_string += f"{minutes} minute{'s' if minutes > 1 else ''}"

    # strip to remove trailing space
    return time_string.strip() 

# shorthand h:m:s formatter
def hms(seconds):
    time_struct = time.gmtime(seconds)
    formatted_time = f"{time_struct.tm_hour}h {time_struct.tm_min}m {time_struct.tm_sec}s"
    return formatted_time

# define output directory
now = datetime.utcnow()  # use UTC time to not leak user location meta
now_hm = now.strftime("%H%M")
now_s = now.strftime("%S")
now_time = now_hm + '-' + now_s + 'Z'
today = now.strftime("%Y-%m-%d")
outputdir = f'output/date_{today}_time_{now_time}'

# define the output files
error_file = os.path.join(outputdir, 'errors.log')
debug_file = os.path.join(outputdir,'debug.log')
summary_file = os.path.join(outputdir, f'summary_{today}.txt')
plot_file = os.path.join(outputdir, f'plot_{today}.png')
output_file = 'sync.log'

# declare the PID of this process
p = psutil.Process(os.getpid())

# Check pirate-cli path is correct
def checkCLIexists(CLI):
    if not os.path.isfile(CLI):
        error_message = f"pirate-cli does not exist at: '{CLI}'"
        err(error_message)
        sys.exit(1)

# read the daemon debug log and stream to queue
def read_log(file, queue, position=0):
    with open(file, 'r') as f:
        # Move to the last known position in the file
        f.seek(position)

        while not startup_complete.is_set():
            line = f.readline().strip()
            if line:
                queue.put(line)
                position = f.tell()  # Update the current position in the file
            else:
                time.sleep(0.1)  # Sleep briefly to avoid busy waiting

# threaded worker for handeling reading the daemon debug.log
def log_monitor(queue):
    log_file_path = os.path.join(datadir, 'debug.log')
    debug('Pirate daemon startup')
    position = os.path.getsize(log_file_path)  # Get the initial position of the file
    read_log(log_file_path, queue, position=position)
    debug('Startup process completed')

# listen for relevant messages in the debug.log and switch cases or set flags
def message_worker(queue, startup_start_time):

    while True:
        line = queue.get(block=True) # block execution until new queue message recieved

        # Bootstrap downloading
        if "init message: Downloading Bootstrap......" in line:
            if not startup_data['downloading_bootstrap']:
                startup_data['downloading_bootstrap'] = True
                startup_data["bootstrap_used"] = True
                startup_data['bootstrap_start_time'] = time.time()
            startup_data["bootstrap_progress"] = float(line.split("......")[-1].strip().strip('%'))
            debug(f'Downloading Bootstrap progress {startup_data["bootstrap_progress"]:.2f}%')

        # Bootstrap downlod finished, now extracting blocks
        elif "init message: Extracting Bootstrap" in line and startup_data['downloading_bootstrap']:     
            startup_data['downloading_bootstrap'] = False
            startup_data['extracting_blocks'] = True
            startup_data["bootstrap_download_time"] = time.time() - startup_data['bootstrap_start_time']
            debug(f"Downloading Bootstrap Complete {hms(startup_data['bootstrap_download_time'])}")
            startup_data['extraction_start_time'] = time.time()   
            debug(f'Extracting Bootstrap Blocks...')  

        # block extraction complete
        # TODO: look for a more reliable signal to mark extraction completion
        elif ("init message: Loading block index" in line or "Block index database configuration" in line) and startup_data['extracting_blocks']:
            startup_data['extracting_blocks'] = False
            startup_data["bootstrap_extraction_time"] = time.time() - startup_data['extraction_start_time']
            debug(f"Bootstrap Block Extraction Complete {hms(startup_data['bootstrap_extraction_time'])}")

        # startup process complete
        elif "init message: Done loading" in line:
            startup_data["startup_time"] = time.time() - startup_start_time
            startup_complete.set()  # Set the exit flag to signal log_monitor worker to exit and the node rpc calls to begin
            break

        # set flag marking rescanning started
        elif "init message: Rescanning" in line and not startup_data['rescanning'] and not startup_data.get("rescan_time"):  
            startup_data["rescan_current_block"] = 0
            startup_data["rescan_progress"] = 0.00 # in case datacollection polls before values are set
            startup_data['rescanning'] = True
            startup_data['rescan_start_time'] = time.time()  
            debug("Rescan Started...")

        # otherwise, log other init messages to the debug.log
        elif "init message:" in line and "Extracting Bootstrap" not in line and "Downloading Bootstrap" not in line:          
            if not startup_data['starting']:
                startup_data['starting'] = True
                daemon_detected.set() # trigger data collection to start
            message = line.split("init message:", 1)[-1].strip()
            debug(f"{message}")

        # if rescanning, collect data
        if "Still rescanning" in line and startup_data['rescanning']:
            startup_data["rescan_current_block"] = int(line.split('block')[1].split('.')[0])
            startup_data["rescan_progress"] = round(float(line.split('=')[-1]) * 100, 2)
            debug(f'rescan progress {startup_data["rescan_progress"]}% (block {startup_data["rescan_current_block"]})')

        # rescanning complete
        if "init message: Activating best chain" in line and startup_data['rescanning']:
            startup_data['rescanning'] = False 
            startup_data["rescan_time"] = time.time() - startup_data['rescan_start_time']
            debug(f"Rescan Complete {hms(startup_data['rescan_time'])}")

        # Building Witness Cache
        #  TODO: checking rescan is mostly complete is a hacky way to prevent the first "setBestChain()" message
        #  from triggering this function, which may or may not appear before the rescan        
        if "SetBestChain()" in line and startup_data["bootstrap_used"] and startup_data["rescan_progress"] > 90: 
            if not startup_data["building_witness"]:
                startup_data["building_witness"] = True
                startup_data["building_witness_start_time"] = time.time() 
                debug(f"Starting Building Witness")
            else:
                startup_data["building_witness"] = False
                startup_data["building_witness_time"] = time.time() - startup_data['building_witness_start_time']
                debug(f"Building Witness Complete {hms(startup_data['building_witness_time'])}")
          
        # log building witness progress
        if "Building Witness" in line and startup_data["building_witness"] and startup_data["bootstrap_used"]:
            if "Building Witnesses for block" in line:
                startup_data["building_witness_block"] = int(line.split('block')[1].split('.')[0])    
                startup_data["building_witness_progress"] = round(float(line.split('=')[-1]) * 100, 2)
                debug(f'rescan progress {startup_data["building_witness_progress"]}% (block {startup_data["building_witness_block"]})')      

    # log in the debug   
    debug("message worker exiting")


# A threaded worker that handles the other workers during the daemon startup process
def startup_worker():
    message_queue = queue.Queue()
    startup_start_time = time.time()

    # start the workers
    threads = [
        Thread(target=log_monitor, args=(message_queue,), name='log_monitor'),
        Thread(target=message_worker, args=(message_queue, startup_start_time), name='message_worker')
    ]
    for thread in threads:
        debug(f"Starting thread: {thread.name}")
        thread.start()

    # wait for them to finish
    for thread in threads:
        thread.join()

    # If we are here the workers have all exited. Startup complete
    debug(f"Total startup time: {hms(startup_data['startup_time'])}")
    startup_data['starting'] = False


# print a welcome graphic
def printSplash():
    # Splash
    msg("\n ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
    msg(" ┃ Pirate Daemon Sync Logger ┃")
    msg(" ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n")


# Create the CSV file where the data samples wil be stored 
def createDataFile():
    # Create a new CSV file with headers
    fileName = f"pirated_synclog_{today}.csv"
    data_file = os.path.join(outputdir, fileName)
    with open(data_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Minutes","Blocks","Memory(GB)","CPU(%)","LoadAvg(1min)","BlockchainSize(GB)","BlocksAdded","Peers","Rescan(Height)","BootstrapDownload(%)","BuildingWitnessCache"])
    msg(f"Output directory: {outputdir}")
    msg(f"CSV file created: {fileName}")
    return data_file

# Main data collection loop that loops until daemon is synced with network
def dataCollectionLoop(start_time, data_file):
    # initialize counters
    prev_blocks = 0
    prev_du_output = None

    # Data Collection loop
    while True:
        # initialize the vars so the write always has something
        blocks = memory = cpu = blockchain_size = block_diff = peers = rescan_block = bootstrap_progress = witnessCache = ''

        # How many minutes since start
        minutes = round((time.time() - start_time) / 60)

        # Format minutes to always display four digits with leading spaces
        formatted_minutes = "{:4d}".format(minutes)
        message = f"{formatted_minutes} min"

        # check for machine data
        try:
            # CPU utilization
            raw_cpu = psutil.cpu_percent(interval=1)
            cpu = "{: >4.1f}".format(raw_cpu)
            load1, load5, load15 = os.getloadavg()
            load1 = "{: >5.2f}".format(load1)
            load5 = "{: >5.2f}".format(load5)
            load15 = "{: >5.2f}".format(load15)

            # Memory utilization in gigabytes
            memory_gb = psutil.virtual_memory().used / (1024 ** 3)

            # Format the memory utilization to always display 3 decimal places
            memory = "{:.3f}".format(memory_gb)

            # Run du command to get blocks directory size in gigabytes
            try:
                # Try to run the du command
                du_output = subprocess.check_output(['du', '-sb', datadir]).decode('utf-8').strip()
                prev_du_output = du_output  # update previous du_output value only if current command succeeds
            except subprocess.CalledProcessError:
                # If it fails, log an error and use the previous output
                err("Error running du command.")
                if prev_du_output is not None:
                    du_output = prev_du_output
                else:
                    du_output = "0"  

            # Extract directory size from the du command output and convert to gigabytes
            blockchain_size_bytes = int(du_output.split()[0])
            blockchain_size_gb = blockchain_size_bytes / (1024 ** 3)

            # Format the size to always display 3 decimal places, including trailing zeros
            blockchain_size = "{:6.3f}".format(blockchain_size_gb)

            # Display Message
            message += f"  │  MEM {memory}GB  CPU {cpu}%  load {load1}  size {blockchain_size}GB"

            # No need trying the RPC until startup is complete
            if startup_complete.is_set():
                try:
                    # get some daemon info via pirate-cli
                    getinfo_output = json.loads(subprocess.check_output([CLI, "getinfo"]).decode('utf-8'))
                    blocks = getinfo_output['blocks']
                    peers = getinfo_output['connections']
                    longestchain = getinfo_output['longestchain']                    
                    block_diff = blocks - prev_blocks

                    # calculate percent complete, but handle zero case
                    if isinstance(longestchain, (int, float)) and longestchain != 0:
                        completed = (blocks / longestchain) * 100
                    else:
                        completed = 0
                    percent_complete = "{:.3f}".format(completed)

                    # set version for the report
                    if not startup_data["PIRATEversion"]:
                        startup_data["PIRATEversion"] = getinfo_output['PIRATEversion']
                    
                    # Update prev_blocks with current blocks
                    prev_blocks = blocks

                    message += f"  │  peers {peers}  sync {percent_complete}%  blocks {blocks} (+{block_diff})"

                    # Check if synced is true
                    synced = getinfo_output['synced']
                    if synced:
                        break

                # if we have an error, log it, but could be not responding while building witnesses
                except subprocess.CalledProcessError as e:
                    if e.returncode == 32:
                        message += f"  │  building witness cache"
                        witnessCache = 1
                    else:
                        message += f"  │  getinfo did not respond (may be busy)"
                        err(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")

            else:
                # handle the cases when getinfo would be blocked
                current_operation = "Daemon starting"
                if startup_data["downloading_bootstrap"]:
                    current_operation = f'Downloading Bootstrap ({startup_data["bootstrap_progress"]:.2f}%)'

                elif startup_data["extracting_blocks"]:
                    current_operation = f'Extracting blocks'

                elif startup_data["building_witness"]:
                    current_operation = f'Building Witness Cache'
                    if startup_data.get("building_witness_progress") and startup_data.get("building_witness_block"):
                        current_operation += f' {"{:.2f}".format(startup_data["building_witness_progress"])}% @ block {startup_data["building_witness_block"]}'

                elif startup_data["rescanning"]:
                    current_operation = f'Rescanning {"{:.2f}".format(startup_data["rescan_progress"])}% @ block {startup_data["rescan_current_block"]}'

                # display the message
                message += f"  │  {current_operation}"

        except Exception as e:
            err(f"Error occurred with machine telemetry: {str(e)}")
            pass

        # print a summary of telemetry each loop
        msg(message)


        if startup_data['rescanning'] and "rescan_current_block" in startup_data:
            rescan_block = startup_data["rescan_current_block"] 

        if startup_data['downloading_bootstrap'] and "bootstrap_progress" in startup_data:
            bootstrap_progress = startup_data["bootstrap_progress"] 

        if startup_data['building_witness']:
            if startup_data.get("building_witness_block"): 
                witnessCache = startup_data["building_witness_block"] 
            else:
                witnessCache = 1

        # Output data to CSV
        with open(data_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([minutes, blocks, memory, cpu, load1, blockchain_size, block_diff, peers, rescan_block, bootstrap_progress, witnessCache])

        # Sleep for the remaining time until the next interval
        sleep_for_interval(start_time)

# detimine building witness time from log
def buildingWitnessCache_minutes(df, sample_rate):
    try:
        # Check if "BuildingWitnessCache" column exists
        if "BuildingWitnessCache" not in df.columns:
            return "unknown"

        # Filter rows where "BuildingWitnessCache" is > 0
        witness_cache = df[df["BuildingWitnessCache"] > 0]["BuildingWitnessCache"]

        # If no rows > 0, return 0
        if len(witness_cache) == 0:
            return 0

        # Sum all the lines that are > 0, then multiply by sample_rate
        total_minutes = int(witness_cache.sum() * sample_rate)

        return total_minutes
    except Exception as e:
        err(f"Error in computing witness cache time: {e}")
        return "unknown"
    
# print the summary to the sync.log and a summary txt file
def write_and_print(f, message):
    msg(message)
    f.write(message + "\n")

# create plot of blocks, and a summary of the sync process
def generateReports(file_path, summary_file, plot_file, bootstrapUsed=startup_data["bootstrap_used"]):
    df = pd.read_csv(file_path)

    # Fill all missing values with 0
    df = df.fillna(0)

    # Fill missing values for certain columns with their means
    for col in ["CPU(%)", "LoadAvg(1min)", "Memory(GB)", "BlockchainSize(GB)"]:
        if col in df.columns:  # Check if column exists
            df[col] = df[col].fillna(df[col].mean())

    # Plot data 
    plt.figure(figsize=(20,10))   
    if bootstrapUsed:
        plt.plot(df["Minutes"], df["Rescan(Height)"], label="Blocks")
        plt.title('Bootstrap Rescan Time')
    else:
        plt.plot(df["Minutes"], df["Blocks"], label="Blocks")
        plt.title('Network Sync Time') 
    plt.xlabel('Time (minutes)')
    plt.ylabel('Height')    
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(plot_file)

    try:
        # set fallbacks in the case of missing data
        column_fallbacks = {
            "CPU(%)": 0,
            "LoadAvg(1min)": 0,
            "Memory(GB)": 0,
            "Minutes": 0,
            "Blocks": 0,
            "BlockchainSize(GB)": 0,
            "Peers": 0
        }

        # Dictionary to hold the computed values
        computed_values = {}

        # compute mean/max data, or use fallbacks for missing datasets
        for column, fallback in column_fallbacks.items():
            if column in df.columns:
                if column == "Minutes":
                    last_minutes_index = df[column].last_valid_index()
                    computed_values[column] = df[column].loc[last_minutes_index]
                else:
                    computed_values[column + "_avg"] = df[column].mean()
                    computed_values[column + "_max"] = df[column].max()
            else:
                computed_values[column + "_avg"] = fallback
                computed_values[column + "_max"] = fallback

        # Handle these with fallbacks in case they don't exist
        cpu_avg = computed_values["CPU(%)_avg"]
        cpu_max = computed_values["CPU(%)_max"]
        load_avg = computed_values["LoadAvg(1min)_avg"]
        load_max = computed_values["LoadAvg(1min)_max"]
        memory_avg = computed_values["Memory(GB)_avg"]
        memory_max = computed_values["Memory(GB)_max"]
        total_minutes = computed_values["Minutes"]
        blocks_synced = int(computed_values["Blocks_max"])
        blockchain_size_total = computed_values["BlockchainSize(GB)_max"]
        avg_peers = int(computed_values["Peers_avg"])

        # define some other environment details
        utc_now = datetime.now(timezone.utc)
        report_date = utc_now.strftime("%A, %d %B %Y, %H:%M:%S %Z%z")
        readable_time = minutes_to_readable_time(total_minutes)
        pirate_version = startup_data['PIRATEversion'] if startup_data.get('PIRATEversion') else "unknown"
        total_mem = psutil.virtual_memory().total / (1024 ** 3) 
        cpu_info = platform.processor() 
        platform_version = platform.platform()
        logical_cores = psutil.cpu_count(logical=True)  # Includes hyper-threading cores
        physical_cores = psutil.cpu_count(logical=False)  # Excludes hyper-threading co>
        cpu_freq = psutil.cpu_freq()

        # set some times
        download_method = "Bootstrap" if bootstrapUsed else "Network Peers"
        bootstrap_download_time = hms(startup_data["bootstrap_download_time"]) if startup_data.get('bootstrap_download_time') else "N/A"
        bootstrap_extraction_time = hms(startup_data["bootstrap_extraction_time"]) if startup_data.get('bootstrap_extraction_time') else "N/A"
        
        
        # display startup time with different resolutions depending on method
        if startup_data.get('startup_time'):
            if startup_data.get('bootstrap_used'):
                startup_time = minutes_to_readable_time(int(startup_data["startup_time"] / 60)) 
            else:
                startup_time = hms(startup_data["startup_time"]) 
        else:
            startup_time = "N/A"

        # decide how to get witness cache time depending on download method
        if startup_data.get("building_witness_time"):
            building_witness_time = hms(startup_data["building_witness_time"])
        else:
            building_witness_time = str(buildingWitnessCache_minutes(df, sample_rate)) + " minutes"

        # fix rescan time
        if startup_data.get("rescan_time") and startup_data.get("building_witness_time"):
            startup_data["rescan_time"] = startup_data["rescan_time"] - startup_data["building_witness_time"]
        
        if startup_data.get("rescan_time"):
            rescan_time = hms(startup_data["rescan_time"])
        else:
            rescan_time = "N/A"

        # see if we can get distro
        try:
            os_info = os.popen('lsb_release -ds').read().strip()
        except Exception as e:
            err(f"Error getting OS info: {str(e)}")
            os_info = "not detected" # Don't think lsb_release works on non linux

        # ok, write the summary to the summary.txt and show in the sync.log
        with open(summary_file, 'w') as f:
            write_and_print(f, "\nPirate Network Sync - Summary Report")
            write_and_print(f, report_date)

            write_and_print(f, "\nENVIRONMENT:")
            write_and_print(f, f"\tpirated: {pirate_version}")
            write_and_print(f, f"\tOS: {os_info}")
            write_and_print(f, f"\tPlatform: {platform_version}")
            write_and_print(f, f"\tMEM: {total_mem:.2f} GB")
            write_and_print(f, f"\tCPU: {cpu_freq.current:.2f}Mhz {cpu_info}")
            write_and_print(f, f"\tCores: {logical_cores} logical | {physical_cores} physical")

            write_and_print(f, "\nNETWORK SYNC DETAILS:")    
            write_and_print(f, f"\tSyncing took {readable_time}")
            write_and_print(f, f"\tBlocks synced: {blocks_synced} ({blockchain_size_total:.2f}GB)")
            write_and_print(f, f"\tPeers: {avg_peers} avg")   
            write_and_print(f, f"\tMEM: {memory_avg:.2f}GB avg ({memory_max:.2f}GB peak)")  
            write_and_print(f, f"\tCPU: {cpu_avg:.2f}% avg ({cpu_max:.2f}% peak)")
            write_and_print(f, f"\tLoad: {load_avg:.2f} avg ({load_max:.2f} peak)")           

            write_and_print(f, "\nDAEMON PROCESSES:")  
            write_and_print(f, f"\tBlock Download Source: {download_method}")
            write_and_print(f, f"\tStartup sequence took {startup_time}")            
            if download_method == "Bootstrap":      
                write_and_print(f, f"\tBootstrap Download {bootstrap_download_time}")   
                write_and_print(f, f"\tBlock Extraction {bootstrap_extraction_time}")
                write_and_print(f, f"\tRescan {rescan_time}")   
            write_and_print(f, f"\tBuilding Witness Cache: {building_witness_time}")                  

        # mark the exit
        msg(f"\nThe sync summary, data CSV, error logs, and chart saved in '{outputdir}'")

    # fuck
    except Exception as e:        
        err(f"An error occurred generating summary report: {str(e)}")

# do any cleanup before exiting
def cleanup():
    nohup_file = 'nohup.out'
    debug("cleaning up, preparing to exit")

    # Move output_file into outputdir
    try:
        if os.path.isfile(output_file):
            shutil.move(output_file, outputdir)
    except Exception as e:
        err(f"An error occurred while moving {output_file} to {outputdir}: {e}")

    # If debug_file exists and is empty, delete it
    try:
        if os.path.isfile(debug_file) and os.path.getsize(debug_file) == 0:
            os.remove(debug_file)
    except Exception as e:
        err(f"An error occurred while deleting {debug_file}: {e}")

    # If nohup.out file was created from launching it in nohup, delete it
    try:
        if os.path.isfile(nohup_file):
            os.remove(nohup_file)
    except Exception as e:
        err(f"An error occurred while deleting {nohup_file}: {e}")
    
    # If error_file exists and is empty, delete it
    try:
        if os.path.isfile(error_file) and os.path.getsize(error_file) == 0:
            os.remove(error_file)
    except Exception as e:
        err(f"An error occurred while deleting {error_file}: {e}")

# main sequence controller
def run():
    # create new output directory
    os.makedirs(outputdir, exist_ok=True)

    # Set up logging for errors and debugging
    configureLogging()

    # check we have access to pirate-cli
    checkCLIexists(CLI)

    # show a welcome message
    printSplash()

    # Start the startup monitoring workers in threads
    startup_thread = Thread(target=startup_worker)
    startup_thread.start()
    debug(f"Active threads after startup: {active_count()}")

    # create data file
    data_file = createDataFile()
    
    # Wait for a node to start
    msg("Listening for node startup\nPlease start Pirate Daemon now")
    daemon_detected.wait()  # Wait here until we see the daemon running

    # We are ready to start collecting data, so start timer and let 'em know! 
    start_time = time.time()
    msg(f"Starting data collection ({sample_rate} minute samples)\n")

    # run the data collection loop until synced = True
    dataCollectionLoop(start_time, data_file)

    # If we are here, Loop finished, pirated in sync with network. Lets finish and get out of here
    total_time = int((time.time() - start_time) / 60)
    msg(f"pirated finished sync'ing with the network. Took {minutes_to_readable_time(total_time)}")

    # Load data from CSV to make a chart and summary
    generateReports(data_file, summary_file, plot_file, startup_data['bootstrap_used'])

    # cleanup
    cleanup()

    # Exit the logger with a success code
    sys.exit(0)

# show user required arguments
def print_commands():
    print("No command provided or unknown command")
    print("The commands are:")
    print("python3 pirated_synclog.py start")
    print("python3 pirated_synclog.py stop")
    print("python3 pirated_synclog.py report <filename> <bootstrap bool>")

# check input arguments
if __name__ == "__main__":
    # Reset the log file
    with open(output_file, 'w') as f:
        pass

    # a argument is required when launching
    if len(sys.argv) > 1:

        # stop
        if sys.argv[1] == 'stop':
            p = subprocess.Popen(['pgrep', '-f', 'pirated_synclog.py'], stdout=subprocess.PIPE)
            out, err = p.communicate()
            for pid in out.splitlines():
                os.kill(int(pid), signal.SIGTERM)
            print("Stopped all running instances of pirated_synclog.py")
            sys.exit(0)

        # report <filename> <bootstrap bool>   
        elif sys.argv[1] == 'report':
            if len(sys.argv) > 3:
                generateReports(sys.argv[2], f'summary_{today}.txt', f'plot_{today}.png', sys.argv[3])
            else:
                print("Data CSV file name is required for report")
                sys.exit(1)

        # start
        elif sys.argv[1] == 'start':
            run()
        else:
            print_commands()
            sys.exit(1)
    else:
        print_commands()
        sys.exit(1)
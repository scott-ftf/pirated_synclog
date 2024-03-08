import os
import sys
import re
import signal
import csv
import time
import subprocess
import json
import logging
import platform
from datetime import datetime, timedelta, timezone
import shutil
import psutil
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
datadir = os.path.join(home, '.komodo/PIRATE')      # location of the daemon datadir (typically /home/$USER/.komodo/PIRATE)
CLI =  os.path.join(home, 'pirate/pirate-cli')      # location of pirate-cli
sample_rate = 1                                     # how many minutes between data collection loops
test_file_size = 100                                # size (mb) of file in mb for testing i/o speed
debug_mode = False                                  # logs more messages to the debug.log

# prepare some flags
startup_data = {
    'rescanning': False,
    'validating_note_position': False,
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
    logger.info("ERROR: " + message)
    message += "\nTraceback:\n"
    message += traceback.format_exc()  # Get traceback information
    logger.error(message)

# Event flag to signal the startup_worker to exit and data collection to start
startup_complete = Event()  
daemon_detected = Event()  

# sleep for the remainder of the loop wait time
def sleep_for_interval(start_time):
    interval = sample_rate * 60
    elapsed_time = time.time() - start_time
    sleep_time = interval - (elapsed_time % interval)
    debug(f"Sleeping for {sleep_time} seconds until next sample")

    # Wait for the remaining time until the next interval
    time.sleep(sleep_time)

# simple function for formatting minutes to days, hours, minutes
def minutes_to_readable_time(total_minutes):
    total_time_td = timedelta(minutes=int(total_minutes))

    days = total_time_td.days
    hours, remainder = divmod(total_time_td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60) 

    time_string = ""
    if days > 0:
        time_string += f"{days} day{'s' if days > 1 else ''} "
    if hours > 0 or days > 0: 
        time_string += f"{hours} hour{'s' if hours > 1 else ''} "
    time_string += f"{minutes} minute{'s' if minutes > 1 else ''}"

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

# Check data directory path is correct
def checkDatadirExists(datadir):
    if not os.path.isdir(datadir):
        error_message = f"The PIRATE data directory does not exist at: '{datadir}'"
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

# threaded worker for handling reading the daemon debug.log
def log_monitor(queue):
    log_file_path = os.path.join(datadir, 'debug.log')
    notified = False
    
    # Continuously check for the existence of the debug.log file
    while not os.path.exists(log_file_path):
        if not notified:
            notified = True
            debug('Waiting for daemon to create debug.log')
        time.sleep(0.1)
    
    debug('Debug.log found, Pirate daemon startup')
    
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

        # if validating note positions, record it
        if "Validating Note Postions..." in line:
            startup_data["validating_note_position"] = True
            startup_data["validating_note_position_progress"] = round(float(line.split('=')[-1]) * 100, 2)
            debug(f'validating note position progress {startup_data["validating_note_position_progress"]}%')

        # if rescanning, collect data
        if "Still rescanning" in line and startup_data['rescanning']:
            startup_data["rescan_current_block"] = int(line.split('block')[1].split('.')[0])
            startup_data["rescan_progress"] = round(float(line.split('=')[-1]) * 100, 2)
            debug(f'rescan progress {startup_data["rescan_progress"]}% (block {startup_data["rescan_current_block"]})')

        # rescanning complete
        rescan_trigger = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}  rescan\s+\d+ms')
        if rescan_trigger.search(line) and startup_data['rescanning']:
            startup_data['rescanning'] = False 
            startup_data["rescan_time"] = time.time() - startup_data['rescan_start_time']
            debug(f"Rescan Complete {hms(startup_data['rescan_time'])}")

        # Initiate building Witness Cache    
        if "Building Witness" in line and not startup_data["building_witness"]: 
                startup_data["building_witness"] = True
                startup_data["building_witness_start_time"] = time.time() 
                debug(f"Starting Building Witness")

          
        # log building witness progress
        if "Building Witness" in line and startup_data["building_witness"]:
            if "Building Witnesses for block" in line:
                startup_data["building_witness_block"] = int(line.split('block')[1].split('.')[0])    
                startup_data["building_witness_progress"] = round(float(line.split('=')[-1]) * 100, 2)
                debug(f'Building Witness progress {startup_data["building_witness_progress"]}% (block {startup_data["building_witness_block"]})')      

            else:
                startup_data["building_witness"] = False
                startup_data["building_witness_time"] = startup_data.get("building_witness_time", 0) + time.time() - startup_data['building_witness_start_time']
                debug(f"Building Witness Complete {hms(startup_data['building_witness_time'])}")

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
        writer.writerow(["Minutes","Blocks","Memory(GB)","CPU(%)","MachineLoadAvg(1min)","BlockchainSize(GB)","BlocksAdded","Peers","Rescan(Height)","ValidateNotePosition","BootstrapDownload(%)","BuildingWitnessCache"])
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
        blocks = memory = cpu = blockchain_size = block_diff = peers = rescan_block = validating_note_position = bootstrap_progress = witnessCache = ''

        # How many minutes since start
        minutes = round((time.time() - start_time) / 60)

        # Format minutes to always display four digits with leading spaces
        formatted_minutes = "{:4d}".format(minutes)
        message = f"{formatted_minutes} min"

        # check for machine data
        try:
            # Total machine load
            load1, load5, load15 = os.getloadavg()
            load1 = "{: >5.2f}".format(load1)
            load5 = "{: >5.2f}".format(load5)
            load15 = "{: >5.2f}".format(load15)

            # Memory and CPU utilization for the pirated process
            memory_gb = 0.000
            cpu_percent = 0.0
            for process in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
                if process.info['name'] == 'pirated':                    
                    # Memory utilization in gigabytes for the process
                    memory_gb = round(process.info['memory_info'].rss / (1024 ** 3), 2)
                   
                    # CPU utilization for the process (measured over 1 second)
                    cpu_percent = round(process.cpu_percent(interval=1))
                    
            memory = "{: >4.2f}".format(memory_gb)
            cpu = "{: >4d}".format(round(cpu_percent))
            
            # Run du command to get blocks directory size in gigabytes
            try:
                # Try to run the du command
                du_output = subprocess.check_output(['du', '-sb', datadir]).decode('utf-8').strip()
                prev_du_output = du_output  # update previous du_output value only if current command succeeds
            except subprocess.CalledProcessError:
                # If it fails, log an error and use the previous output
                err("Error running du command. Using previous value and continuing.")
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
            message += f" │ pirated: {memory}GB MEM  {cpu}% CPU │ machine: {load1} load  {blockchain_size}GB hdd"

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
                    message += f" │ peers {peers}  sync {percent_complete}%  blocks {blocks} (+{block_diff})"

                    # Check if synced is true
                    synced = getinfo_output['synced']
                    if synced:
                        break

                # if we have an error, log it, but could be not responding while building witnesses
                except subprocess.CalledProcessError as e:
                    if e.returncode == 32:
                        message += f" │ building witness cache"
                        witnessCache = 1
                    else:
                        message += f" │ getinfo did not respond (may be busy)"
                        err(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}.")

            else:
                # handle the cases when getinfo would be blocked
                current_operation = "Waiting for daemon to complete task..."
                if startup_data["downloading_bootstrap"]:
                    current_operation = f'Downloading Bootstrap ({startup_data["bootstrap_progress"]:.2f}%)'

                elif startup_data["extracting_blocks"]:
                    current_operation = f'Extracting blocks'

                elif startup_data["building_witness"]:
                    current_operation = f'Building Witness Cache'
                    if startup_data.get("building_witness_progress") and startup_data.get("building_witness_block"):
                        current_operation += f' {"{:.2f}".format(startup_data["building_witness_progress"])}% @ block {startup_data["building_witness_block"]}'

                elif startup_data["validating_note_position"]:
                    current_operation = f'Validating Note Position: {"{:.2f}".format(startup_data["validating_note_position_progress"])}%'

                elif startup_data["rescanning"]:
                    current_operation = f'Rescanning {"{:.2f}".format(startup_data["rescan_progress"])}% @ block {startup_data["rescan_current_block"]}'

                # display the message
                message += f" │ {current_operation}"

        except Exception as e:
            err(f"Error occurred with machine telemetry: {str(e)}")
            pass

        # print a summary of telemetry each loop
        msg(message)

        if startup_data["validating_note_position"]:
            validating_note_position = startup_data["validating_note_position_progress"]
            startup_data["validating_note_position"] = False

        if startup_data['rescanning'] and "rescan_current_block" in startup_data:
            rescan_block = startup_data["rescan_current_block"] 

        if startup_data['downloading_bootstrap'] and "bootstrap_progress" in startup_data:
            bootstrap_progress = startup_data["bootstrap_progress"] 

        if startup_data['building_witness']:
            if startup_data.get("building_witness_block"): 
                witnessCache = startup_data["building_witness_block"] 
            else:
                witnessCache = 1

        # write to the output csv each loop
        with open(data_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([minutes, blocks, memory_gb, cpu_percent, load1, blockchain_size, block_diff, peers, rescan_block, validating_note_position, bootstrap_progress, witnessCache])

        # wait for the remainder of the loop time
        sleep_for_interval(start_time)

# detimine building witness time from log
def buildingWitnessCache_minutes(df, sample_rate):
    try:
        if "BuildingWitnessCache" not in df.columns:
            return "unknown"

        # Filter rows where "BuildingWitnessCache" is > 0
        witness_cache = df[df["BuildingWitnessCache"] > 0]["BuildingWitnessCache"]
        if len(witness_cache) == 0:
            return 0
        total_minutes = int(witness_cache.sum() * sample_rate)

        return total_minutes
    except Exception as e:
        err(f"Error in computing witness cache time: {e}")
        return "unknown"

# Return the total and available storage size  
def get_storage_info(path):
    try:
        stat = os.statvfs(path)
        total_space = stat.f_frsize * stat.f_blocks  
        total_storage = total_space / (1024**3) 
        available_space = stat.f_frsize * stat.f_bavail 
        available_storage = available_space / (1024**3)
        return total_storage, available_storage
    except Exception as e:
        return "?", "?"
import os
import time

# Test disk write speed
def measure_write_speed():
    block_size = 1024 * 1024  # 1 MB
    blocks = int(test_file_size)
    buffer = os.urandom(block_size)
    test_file_path = os.path.join(datadir, 'test_file')
    
    try:
        start_time = time.time()
        
        with open(test_file_path, 'wb') as f:
            for _ in range(blocks):
                f.write(buffer)

        end_time = time.time()
        os.remove(test_file_path)  # Clean up
        write_speed = test_file_size / (end_time - start_time)
        return f"{write_speed:.2f} MB/s"
    
    except:
        return "unknown"

# Test disk read speed
def measure_read_speed():
    block_size = 1024 * 1024  # 1 MB
    blocks = int(test_file_size)
    buffer = os.urandom(block_size)
    test_file_path = os.path.join(datadir, 'test_file')
    
    try:
        # First, create the file to be read
        with open(test_file_path, 'wb') as f:
            for _ in range(blocks):
                f.write(buffer)
        
        start_time = time.time()

        with open(test_file_path, 'rb') as f:
            while f.read(block_size):
                pass

        end_time = time.time()
        os.remove(test_file_path)  # Clean up
        read_speed = test_file_size / (end_time - start_time)
        return f"{read_speed:.2f} MB/s"

    except:
        return "unknown"
    
def get_cpu_name_linux():
    with open('/proc/cpuinfo') as f:
        for line in f:
            if 'model name' in line:
                return line.partition(':')[2].strip()
    return "Unknown"

def get_physical_cpus_linux():
    result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, text=True)
    output = result.stdout
    for line in output.splitlines():
        if 'Socket(s):' in line:
            return int(line.split(':')[1].strip())
    return 0

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
    for col in ["CPU(%)", "MachineLoadAvg(1min)", "Memory(GB)", "BlockchainSize(GB)"]:
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
            "MachineLoadAvg(1min)": 0,
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
        cpu_avg = int(computed_values["CPU(%)_avg"])
        cpu_max = int(computed_values["CPU(%)_max"])
        load_avg = computed_values["MachineLoadAvg(1min)_avg"]
        load_max = computed_values["MachineLoadAvg(1min)_max"]
        memory_avg = computed_values["Memory(GB)_avg"]
        memory_max = computed_values["Memory(GB)_max"]
        total_minutes = computed_values["Minutes"]
        blocks_synced = int(computed_values["Blocks_max"])
        blockchain_size_max = computed_values["BlockchainSize(GB)_max"]
        avg_peers = int(computed_values["Peers_avg"])
        max_peers = int(computed_values["Peers_max"])

        # Get some wallet info after sync
        def safe_subprocess_call(command):
            try:
                output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode('utf-8')
                return json.loads(output)
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                return "unknown"
            
        walletinfo = safe_subprocess_call([CLI, "getwalletinfo"]) 
        networkinfo = safe_subprocess_call([CLI, "getnetworkinfo"])

        # evaluate wallet data
        txcount = walletinfo["txcount"]
        saplingnotes = walletinfo["saplingnotes"]
        arctxcount = walletinfo["arctxcount"]
        arcsaplingnotes = walletinfo["arcsaplingnotes"]
        saplingaddresses = walletinfo["saplingaddresses"]
        saplingspendingkeys = walletinfo["saplingspendingkeys"]
        saplingfullviewingkeys = walletinfo["saplingfullviewingkeys"]
        keypoololdest = datetime.utcfromtimestamp(walletinfo["keypoololdest"])
        keypoolsize = walletinfo["keypoolsize"]
        
        # Evaluate the network infos
        reachable = []
        unreachable = []
        if isinstance(networkinfo, dict) and "networks" in networkinfo:
            for network in networkinfo["networks"]:
                if network["reachable"] and network["name"]:
                    reachable.append(network["name"])
                elif network["name"]:  # We only want to list named networks
                    unreachable.append(network["name"])

        # define some other environment details
        utc_now = datetime.now(timezone.utc)
        report_date = utc_now.strftime("%A, %d %B %Y, %H:%M:%S %Z%z")
        readable_time = minutes_to_readable_time(total_minutes)
        sync_time = hms(total_minutes * 60)
        pirate_version = startup_data['PIRATEversion'] if startup_data.get('PIRATEversion') else "unknown"
        root_type = os.popen(f'df -T {datadir} | awk \'NR==2 {{print $2}}\'').read().strip() or "Unknown"
        total_mem = psutil.virtual_memory().total / (1024 ** 3)        
        platform_version = platform.platform()
        logical_cores = psutil.cpu_count(logical=True)  # Includes hyper-threading cores
        physical_cores = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        cpu_info = platform.processor() 
        processor_name = get_cpu_name_linux()  
        num_physical_cpus = get_physical_cpus_linux()
        total_storage, available_storage = get_storage_info(datadir) 
        write_speed = measure_write_speed()
        read_speed = measure_read_speed()

        # set some times
        download_method = "Bootstrap" if bootstrapUsed else "Network Peers"
        bootstrap_download_time = hms(startup_data["bootstrap_download_time"]) if startup_data.get('bootstrap_download_time') else "N/A"
        bootstrap_extraction_time = hms(startup_data["bootstrap_extraction_time"]) if startup_data.get('bootstrap_extraction_time') else "N/A"
        
        # Get final blockchain size
        du_output = subprocess.check_output(['du', '-sb', datadir + "/blocks"]).decode('utf-8').strip()
        blockchain_size_bytes = int(du_output.split()[0])
        blockchain_size_gb = blockchain_size_bytes / (1024 ** 3)            

        # display startup time with different resolutions depending on method
        if startup_data.get('startup_time'):
            startup_time = hms(startup_data["startup_time"]) 
        else:
            startup_time = "UNKOWN"

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

        # detimine time spend validating note position
        note_position_row_count = df['ValidateNotePosition'].notnull().sum()
        validating_note_position_mins = sample_rate * note_position_row_count
        validating_note_position_time = hms(validating_note_position_mins * 60)

        # see if we can get distro
        try:
            os_info = os.popen('lsb_release -ds').read().strip()
        except Exception as e:
            err(f"Error getting OS info: {str(e)}")
            os_info = "not detected" # Don't think lsb_release works on non linux

        # ok, write the summary to the summary.txt and show in the sync.log
        with open(summary_file, 'w') as f:
            write_and_print(f, "\nPIRATE NETWORK SYNC - Summary Report")

            write_and_print(f, "\n\nSYNCHRONIZATION TIME:")
            write_and_print(f, f"\t{readable_time}")
            write_and_print(f, f"\t{report_date}")

            write_and_print(f, "\nENVIRONMENT:")
            write_and_print(f, f"\tOperating system:       {os_info}")
            write_and_print(f, f"\tPlatform:               {platform_version}") 
            write_and_print(f, f"\tFile system:            {root_type}")
            write_and_print(f, f"\tDisk space free:        {available_storage:.2f} GB of {total_storage:.2f} GB")
            write_and_print(f, f"\tRead speed test:        {read_speed}")
            write_and_print(f, f"\tWrite speed test:       {write_speed}")
            write_and_print(f, f"\tMemory:                 {total_mem:.2f} GB")
            write_and_print(f, f"\tProcessor Name:         {processor_name}")  
            write_and_print(f, f"\tArchitecture:           {cpu_info}")
            write_and_print(f, f"\tFrequency:              {cpu_freq.max:.2f}Mhz")   
            write_and_print(f, f"\tPhysical CPUs:          {num_physical_cpus}")
            write_and_print(f, f"\tCores:                  {logical_cores} logical | {physical_cores} physical")

            write_and_print(f, "\nNODE DETAILS:")
            write_and_print(f, f"\tPirate daemon version:  {pirate_version}")            
            write_and_print(f, f"\tReachable networks:     {len(reachable)}" + (f" ({', '.join(reachable)})" if reachable else ""))
            write_and_print(f, f"\tUnreachable networks:   {len(unreachable)}" + (f" ({', '.join(unreachable)})" if unreachable else ""))
            write_and_print(f, f"\tTx count:               {txcount}")  
            write_and_print(f, f"\tSapling notes:          {saplingnotes}")
            write_and_print(f, f"\tArchived tx count:      {arctxcount}")
            write_and_print(f, f"\tArchived sapling notes: {arcsaplingnotes}")
            write_and_print(f, f"\tSapling addresses:      {saplingaddresses}")
            write_and_print(f, f"\tSapling spending keys:  {saplingspendingkeys}")
            write_and_print(f, f"\tSapling full view keys: {saplingfullviewingkeys}")
            write_and_print(f, f"\tKey pool oldest:        {keypoololdest}")
            write_and_print(f, f"\tKey pool size:          {keypoolsize}")       

            write_and_print(f, "\nTELEMETRY SUMMARY:")   
            write_and_print(f, f"\tBlock download source:  {download_method}") 
            write_and_print(f, f"\tBlocks synced:          {blocks_synced}")
            write_and_print(f, f"\tBlockchain size:        {blockchain_size_gb:.2f}GB")            
            write_and_print(f, f"\tMax disk used:          {blockchain_size_max:.2f}GB")
            if download_method != "Bootstrap": 
                write_and_print(f, f"\tPeers:                  {avg_peers} avg ({max_peers} peak)")   
            write_and_print(f, f"\tPirated MEM:            {memory_avg:.2f}GB avg ({memory_max:.2f}GB peak)")  
            write_and_print(f, f"\tPirated CPU:            {round(cpu_avg):d}% avg ({round(cpu_max):d}% peak)")
            write_and_print(f, f"\tMachine load:           {load_avg:.2f} avg ({load_max:.2f} peak)")           

            write_and_print(f, "\nSYNC PROCESSES:")  
            write_and_print(f, f"\tTotal sync time:        {readable_time}")            
            write_and_print(f, f"\tStartup sequence:       {startup_time}")            
            if download_method == "Bootstrap":      
                write_and_print(f, f"\tBootstrap download:     {bootstrap_download_time}")   
                write_and_print(f, f"\tBlock extraction:       {bootstrap_extraction_time}")
                write_and_print(f, f"\tRescan:                 {rescan_time}")   
            write_and_print(f, f"\tBuilding witness cache: {building_witness_time}")   
            write_and_print(f, f"\tValidate note position: {validating_note_position_time}")              

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
            shutil.copy2(output_file, outputdir)
    except Exception as e:
        err(f"An error occurred while copying {output_file} to {outputdir}: {e}")

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

    # check we have access to pirate-cli, and that the data directory exists
    checkCLIexists(CLI)
    checkDatadirExists(datadir)

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
    msg(f"pirated syncronized with the network.")
    msg(f"Took {minutes_to_readable_time(total_time)}")
    msg(f"preparing report...")

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
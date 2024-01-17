let chart;
let currentFileName = '';
let currentFileContent = '';

// Define your color mapping
const colorMapping = {
    "Blocks": '#2d2d2d',
    "Memory(GB)": '#8e00ff',
    "CPU(%)": '#ee17ed',
    "BlockchainSize(GB)": '#40cbd6',
    "BlocksAdded": '#329542',
    "Peers": '#c9ff75',
    "Rescan(Height)": '#d13278',
    "BootstrapDownload(%)": '#499bf7',
    "MachineLoadAvg(1min)": "#ffffff",
    "BuildingWitnessCache": "#ee9836"
};

// Load array to store the names of the benchmark files
let benchmarkFiles = {};

window.onload = function() {
    loadBenchmarkFiles();
};

function loadBenchmarkFiles() {
    fetch('./benchmarkFiles.json')
    .then(response => response.json())
    .then(data => {
        benchmarkFiles = data;        
        populateVersions();
        checkUrlAndLoadFile();
    })
    .catch(error => console.error('Error loading benchmark files:', error));
}

// populate the versions dropdown with all posible versions
function populateVersions() {
    let versionSelect = document.getElementById('versionSelect');
    versionSelect.appendChild(new Option('Select Version', ''));

    for (let version in benchmarkFiles) {
        let option = document.createElement('option');
        option.value = version;
        option.text = version;
        versionSelect.appendChild(option);
    }
    updateBenchmarkButtonVisibility();
}

// populate source options if version is selected
function populateSources() {
    let version = document.getElementById('versionSelect').value;
    let sourceSelect = document.getElementById('sourceSelect');
    let sourceDiv = document.getElementById('sourceDiv');
    let benchmarkSelect = document.getElementById('benchmarkSelect');
    let benchmarkDiv = document.getElementById('benchmarkDiv');
    let uploader = document.getElementById('uploader');

    if (version === '') {
        sourceDiv.style.display = 'none';
        benchmarkDiv.style.display = 'none';
        uploader.style.display = 'block';
        sourceSelect.innerHTML = '';
        benchmarkSelect.innerHTML = '';
    } else {
        sourceSelect.innerHTML = '';
        sourceSelect.appendChild(new Option('Select Source', ''));
        for (let source in benchmarkFiles[version]) {
            let option = document.createElement('option');
            option.value = source;
            option.text = source;
            sourceSelect.appendChild(option);
        }
        sourceDiv.style.display = 'flex';
        uploader.style.display = "none";
    }
    updateBenchmarkButtonVisibility(); 
}

// populate benchmak options if version and source are selected
function populateBenchmarks() {
    let version = document.getElementById('versionSelect').value;
    let source = document.getElementById('sourceSelect').value;
    let benchmarkSelect = document.getElementById('benchmarkSelect');
    let benchmarkDiv = document.getElementById('benchmarkDiv');

    benchmarkSelect.innerHTML = '';
    benchmarkSelect.appendChild(new Option('Select Benchmark', ''));
    for (let benchmark in benchmarkFiles[version][source]) {
        let option = document.createElement('option');
        option.value = benchmark;
        option.text = benchmark;
        benchmarkSelect.appendChild(option);
    }
    benchmarkDiv.style.display = 'flex';
    updateBenchmarkButtonVisibility(); 
}

// Make load button vilible, only when all three options are selected
function updateBenchmarkButtonVisibility() {
    let version = document.getElementById('versionSelect').value;
    let source = document.getElementById('sourceSelect').value;
    let benchmark = document.getElementById('benchmarkSelect').value;
    let benchmarkBtn = document.getElementById('benchmarkBtn');

    if (version && source && benchmark && version !== '' && source !== '' && benchmark !== '') {
        benchmarkBtn.style.display = 'block';
    } else {       
        benchmarkBtn.style.display = 'none';
    }
}

// When button is clicked, check all options before loading
function loadSelectedFile() {
    let version = document.getElementById('versionSelect').value;
    let source = document.getElementById('sourceSelect').value;
    let benchmark = document.getElementById('benchmarkSelect').value;

    if (version && source && benchmark && version !== '' && source !== '' && benchmark !== '') {
        loadFile(version, source, benchmark);
    } else {
        alert("Please select all options before loading the benchmark.");
    }    
}

// load the text file and append its contents to a div
function loadTextFile(file) {
    return fetch(file)
        .then(response => {
            if (response.ok) {
                return response.text();
            } else {
                throw new Error('File not found');
            }
        })
        .then(contents => {
            // Check if "extra_data_textbox" exists
            let extraDataTextbox = document.getElementById("extra_data_textbox");
            
            // If it doesn't exist, create a new one
            if (!extraDataTextbox) {
                extraDataTextbox = document.createElement("pre");
                extraDataTextbox.id = "extra_data_textbox";
                document.getElementById("report").appendChild(extraDataTextbox);
            }
            
            // Update the contents
            extraDataTextbox.textContent = contents;
        })
        .catch(error => {
            // If the file is not found, do nothing (only solution for client side)
        });
}

// load an image file and append it to a div
function loadImage(file) {
    return fetch(file)
        .then(response => {
            if (response.ok) {
                // Check if "extra_data_imgbox" exists
                let extraDataImgbox = document.getElementById("extra_data_imgbox");

                // If it doesn't exist, create a new one
                if (!extraDataImgbox) {
                    extraDataImgbox = document.createElement("div");
                    extraDataImgbox.id = "extra_data_imgbox";
                    document.getElementById("report").appendChild(extraDataImgbox);
                }

                // Remove existing content
                extraDataImgbox.innerHTML = '';

                let img = document.createElement("img");
                img.src = file;

                // Create anchor element
                let anchor = document.createElement("a");
                // Point the anchor to the image source
                anchor.href = file;
                // Set the anchor target to '_blank' to open in a new tab
                anchor.target = '_blank';
                // Append the img to the anchor
                anchor.appendChild(img);

                // Append the anchor to the div instead of the image
                extraDataImgbox.appendChild(anchor);
            } else {
                throw new Error('Image not found');
            }
        })
        .catch(error => {
            // If the image file is not found, do nothing
        });
}

function createDownloadButton(filePath) {
    // Check if a button with the id "downloadButton" already exists
    var existingButton = document.getElementById("downloadButton");
    let fileName = filePath.split("/").pop();


    // If it does exist, remove it
    if (existingButton) {
        existingButton.parentNode.removeChild(existingButton);
    }

    // Create new button element
    var btn = document.createElement("a");

    // Set button attributes
    btn.id = "downloadButton";
    btn.href = filePath;
    btn.title = "download this benchmark CSV data";
    btn.download = filePath.split("/").pop()
    btn.innerText = '⬇ ' + fileName;

    // Append the button to the "report" element
    document.getElementById("extra_data").appendChild(btn);
}

// Handles the file input operation
function handleFile() {
    const fileInput = document.getElementById('csvFile');            
    const file = fileInput.files[0];
    const reader = new FileReader();

    // Update the current file's name
    currentFileName = file.name;
    //document.getElementById("downloadButton").style.display = 'none';

    // Parsing the content of the file
    reader.onload = function (e) {
        const contents = e.target.result;
        currentFileContent = contents; // Save the current file content
        const data = parseCSV(contents);        
        renderChart(data, currentFileName);
        document.getElementById("extra_data").style.display = "none";
    };
    reader.readAsText(file);
}

function checkUrlAndLoadFile() {
    const params = new URLSearchParams(window.location.search);
    const version = params.get('version');
    const source = params.get('source');
    const benchmark = params.get('benchmark');
    if (version && source && benchmark) {
        loadFile(version, source, benchmark);
    }
}

function updateUrlWithParameters(version, source, benchmark) {
    const params = new URLSearchParams();
    params.set('version', version);
    params.set('source', source);
    params.set('benchmark', benchmark);

    const newUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
    window.history.pushState({}, '', newUrl);
}

// handle the file loading operation
function loadFile(version, source, benchmark) {
    var title = version + " " + source + " sync (" + benchmark + ")" 
    document.getElementById("headline").innerHTML = title;

    let request = new XMLHttpRequest();

    let filePath = `./benchmarks/${version}/${source}/${benchmark}/`;

    let csvFileName = benchmarkFiles[version][source][benchmark];

    request.open("GET", filePath + csvFileName, true);
    request.onload = function(e) {
        const contents = e.target.responseText;
        currentFileContent = contents; // Save the current file content
        const data = parseCSV(contents);        
        renderChart(data, csvFileName);
    };
    request.send();

    updateUrlWithParameters(version, source, benchmark); //add the selected options to the url

    // check for corresponding .txt file
    let txtFileName = csvFileName.replace('.csv', '.txt');
    let textFilePromise = loadTextFile(filePath + txtFileName);

    // After loading .txt, check for corresponding .png file
    let pngFileName = csvFileName.replace('.csv', '.png');
    let imageFilePromise = loadImage(filePath + pngFileName);

    // Wait for both promises to complete before creating the download button
    Promise.all([textFilePromise, imageFilePromise]).then(() => {
        createDownloadButton(filePath + csvFileName);
        });     
}

function parseCSV(text) {
    const lines = text.split('\n');
    const headers = lines[0].split(',').map(header => header.trim());  // trim headers
    let lastValidCPU = null; // Initialize last valid CPU value
    const data = lines.slice(1).map(line => {
        const values = line.split(',');
        let row = {};
        headers.forEach((header, index) => {
            // replace zero values in CPU% with the previous valid value 
            if (header === 'CPU(%)') { // Check if current header is CPU(%)
                if (values[index] !== '' && parseFloat(values[index]) !== 0) { // If value is not empty or zero, it's valid
                    lastValidCPU = values[index]; // Update last valid CPU value
                }
                row[header] = lastValidCPU; // Use last valid CPU value for current row
            } else {
                // If value exists at this index, trim it
                if (values[index] !== undefined) {
                    row[header] = values[index].trim();  // trim values
                }
            }
        });
        return row;
    });
    return data;
}

// destroy the previous chart if any
function destroyChart() {
    if (chart) {
        chart.destroy();
    }
}

// prepare labels and datasets for the chart
function prepareData(data, fileName) {
    const labels = data.map(row => row.Minutes);
    const datasets = [];
    const colors = [];

    let colorIndex = 0;

    for (let header in data[0]) {
        header = header.trim();
        if (header !== 'Minutes') {
            const values = data.map(row => Number(String(row[header]).trim()));
            let color;

            if (colorMapping[header]) {
                color = parseHexColor(colorMapping[header]);
            } else {
                console.warn(`Color for header "${header}" is not defined in colorMapping. Using a random color.`);
                color = generateRandomColor();
            }

            const dataset = createDataset(header, values, color, fileName);
            datasets.push(dataset);
            colors.push(color);
            colorIndex++;
        }
    }

    return { labels, datasets, colors };
}

// parse a hex color
function parseHexColor(hexColor) {
    return {
        r: parseInt(hexColor.slice(1, 3), 16),
        g: parseInt(hexColor.slice(3, 5), 16),
        b: parseInt(hexColor.slice(5, 7), 16),
    };
}

// generate a random color
function generateRandomColor() {
    return {
        r: Math.floor(Math.random() * 256),
        g: Math.floor(Math.random() * 256),
        b: Math.floor(Math.random() * 256),
    };
}

//  create a dataset for each header
function createDataset(header, values, color, fileName) {
    let dataset = {
        label: header,
        data: values,
        borderColor: `rgba(${color.r}, ${color.g}, ${color.b}, 0.6)`,
        pointBackgroundColor: 'rgba(0, 0, 0, 0)',
        pointBorderWidth: 0,
        backgroundColor: `rgba(${color.r}, ${color.g}, ${color.b})`,
        fill: false,
        yAxisID: header,
        pointRadius: 14,
        pointHoverRadius: 20,
        borderWidth: 2,
        hoverBorderWidth: 4,
        order: 1,
        tension: 0.2,
        hidden: isHeaderHidden(header, fileName),
    };

    if (header === 'Blocks') {
        dataset = setSpecialPropertiesForBlocks(dataset);
    } else if (header === "Rescan(Height)" || header === "BootstrapDownload(%)") {
        dataset = setSpecialPropertiesForRescanBootstrap(dataset, color);
    } else if (header === "BuildingWitnessCache") {
        dataset = setSpecialPropertiesWitness(dataset, color);
    }

    return dataset;
}

// check if a header should be hidden
function isHeaderHidden(header, fileName) {
    return (
        header === "MachineLoadAvg(1min)" ||
        ((header === "BootstrapDownload(%)" || header === "Rescan(Height)") && !fileName.toUpperCase().includes("BOOTSTRAP")) ||
        ((header === "BlockchainSize(GB)" || header === "BlocksAdded" || header === "Peers") && fileName.toUpperCase().includes("BOOTSTRAP"))
    );
}

//  set special properties for 'Blocks' dataset
function setSpecialPropertiesForBlocks(dataset) {
    dataset.pointRadius = 30;
    dataset.pointHoverRadius = 16;
    dataset.pointStyle = 'rect';
    dataset.borderWidth = 1;
    dataset.borderColor = '#959595';
    dataset.backgroundColor = '#2d2d2d';
    dataset.pointBorderColor = '#545454';
    dataset.pointBackgroundColor = 'rgba(0, 0, 0, 0)',
    dataset.fill = 'origin';
    dataset.tension = 0.4;
    dataset.order = 2;

    return dataset;
}

// set special properties for 'Rescan(Height)' and 'BootstrapDownload(%)' datasets
function setSpecialPropertiesForRescanBootstrap(dataset, color) {
    dataset.borderColor = `rgba(${color.r}, ${color.g}, ${color.b})`;
    dataset.backgroundColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.04)`;
    dataset.pointBorderColor = `rgba(${color.r}, ${color.g}, ${color.b})`;
    dataset.pointBackgroundColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.01)`;
    dataset.fill = 'origin';
    dataset.tension = 0.4;
    dataset.order = 2;

    return dataset;
}

function setSpecialPropertiesWitness(dataset, color) {
    dataset.borderColor = 'rgba(0, 0, 0, 0)';
    dataset.backgroundColor = `rgba(${color.r}, ${color.g}, ${color.b}, 0.1)`;
    dataset.pointBorderColor = 'rgba(0, 0, 0, 0)';
    dataset.pointBackgroundColor = 'rgba(0, 0, 0, 0)';
    dataset.fill = 'origin';
    dataset.tension = 0.4;
    dataset.order = 3;

    return dataset;
}

// assign each dataset a separate Y-axis
function assignYAxes(datasets, colors) {
    for (let i = 0; i < datasets.length; i++) {
        const dataset = datasets[i];
        const color = colors[i];

        const yAxisID = dataset.yAxisID;

        chart.options.scales[yAxisID] = {
            type: 'linear',
            display: true,
            position: yAxisID === 'Blocks' ? 'left' : 'right',
            beginAtZero: true,
            grid: { display: false },
            ticks: {
                display: yAxisID === 'Blocks' || yAxisID === 'BlockchainSize(GB)' || yAxisID === 'CPU(%)',
                fontColor: color,
            },
            title: {
                display: yAxisID === 'Blocks' || yAxisID === 'BlockchainSize(GB)' || yAxisID === 'CPU(%)',
                text: yAxisID,
            },
        };

        dataset.yAxisID = yAxisID;
    }
}

//  customize tooltip callbacks
function customizeTooltip() {
    Chart.defaults.plugins.tooltip.callbacks.title = function (context) {
        if (context.length > 0) {
            return 'Sync Minutes: ' + context[0].label;
        }
        return '';
    };
}

// update the chart and apply changes
function updateChart() {
    var controls = document.getElementById('controls');
    controls.style.opacity = "0";
    controls.style.width = '0px';
    controls.style.maxHeight = '0px';
    controls.style.marginBottom = '80px';

    var reloadButton = document.getElementById('reloadButton')
    reloadButton.style.opacity = 1;
    reloadButton.style.visibility = 'visible';

    document.getElementById("canvas_container").style.display = 'block';
    document.getElementById("art").style.width = '30px';
    chart.update();
    document.getElementById("extra_data").style.display = 'block';
}

// render the chart with the parsed CSV data
function renderChart(data, fileName) {
    const chartContainer = document.getElementById('chart');

    destroyChart();

    const { labels, datasets, colors } = prepareData(data, fileName);
    const ctx = chartContainer.getContext('2d');

    chart = new Chart(ctx, {
        type: 'line',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Pirate Synclog Data' },
                legend: {
                    labels: {
                        generateLabels: function (chart) {
                            const dataset = chart.data.datasets[0];
                            const labels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                            labels.forEach(label => {
                                if (label.text === dataset.label) {
                                    label.fillStyle = dataset.borderColor;
                                    label.strokeStyle = dataset.borderColor;
                                }
                            });
                            return labels;
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        beforeBody: function (tooltipItems) {
                            chart.tooltipColor = chart.data.datasets[tooltipItems[0].datasetIndex].borderColor;
                        },
                        labelColor: function (context) {
                            return {
                                borderColor: 'rgb(0, 0, 0)',
                                backgroundColor: chart.tooltipColor,
                            };
                        },
                        label: function (context) {
                            const lines = [];
                            let label = context.dataset.label || '';

                            if (context.parsed.y !== null) {
                                label += `: ${context.parsed.y}`;
                            }
                            lines.push(label);

                            // Check if we are not currently hovering over the "Blocks" dataset
                            if (context.dataset.label !== 'Blocks') {
                                // Get the "Blocks" dataset
                                const blocksDataset = chart.data.datasets.find(dataset => dataset.label === 'Blocks');
                                if (blocksDataset) {
                                    // Get the corresponding "Blocks" value
                                    const blocksValue = blocksDataset.data[context.parsed.x]; // assuming x values are indices
                                    // Only add "Blocks" value if it is greater than 0
                                    if (blocksValue > 0) {
                                        // Add the "Blocks" value to the label
                                        lines.push(`@ height: ${blocksValue}`);
                                    }
                                }
                            }

                            return lines;
                        }
                    }
                }

            },
            scales: {
                x: { display: true, title: { display: true, text: 'Sync Minutes' } }
            }
        }
    });

    assignYAxes(datasets, colors);
    customizeTooltip();
    updateChart();
}

function loadMainPage() {
    const baseUrl = window.location.origin + window.location.pathname;
    window.history.pushState({}, '', baseUrl);
    window.location.reload(); 
}
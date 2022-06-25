"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.doFullReads = exports.initializeOBSLogReader = exports.handleNewFile = void 0;
const chokidar = __importStar(require("chokidar"));
const fs = __importStar(require("fs"));
const tail_file_1 = __importDefault(require("@logdna/tail-file"));
const sound = require("sound-play");
const uploadNft_1 = __importDefault(require("./uploadNft"));
let tail;
let filesList = [];
let doFullRead = false;
let window;
function handleNewFile(filePath) {
    console.log('new file: ', filePath);
    sound.play('assets/save.mp3');
    window.webContents.send('new-video', {
        filePath: filePath
    });
    (0, uploadNft_1.default)(filePath);
}
exports.handleNewFile = handleNewFile;
function parseChunk(chunk) {
    let lines = chunk.split('\n');
    let splitLine;
    let timeStamp;
    let message;
    let events = [];
    for (let line of lines) {
        if (line.search('[Debug]')) {
            splitLine = line.split('[Debug]');
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message });
        }
        else if (line.search('[Info]')) {
            splitLine = line.split('[Info]');
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message });
        }
        else if (line.search('[Warning]')) {
            splitLine = line.split('[Warning]');
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message });
        }
        else if (line.search('[Warning]')) {
            splitLine = line.split('[Warning]');
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message });
        }
        else if (line.search('[Error]')) {
            splitLine = line.split('[Error]');
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message });
        }
        else if (line.search('[]')) { }
        if (events[events.length - 1]['message'].indexOf("Wrote replay buffer to ") !== -1) {
        }
    }
    return events;
}
function fireEvent(eventName, data) {
    if (eventName === 'newFile') {
        console.log('new file');
        handleNewFile(data);
    }
}
function handleEvents(events) {
    for (let event of events) {
        if (event['message'].indexOf("Wrote replay buffer to ") !== -1) {
            let fileName = event['message'].split("Wrote replay buffer to ")[1];
            // fileName = fileName.slice(1, -2);
            fileName = fileName.slice(1, -1);
            fireEvent('newFile', fileName);
        }
    }
}
function startTail(filePath) {
    const newTail = new tail_file_1.default(filePath, { encoding: 'utf8' });
    newTail.on('data', (chunk) => {
        let events = parseChunk(chunk);
        handleEvents(events);
    })
        .on('tail_error', (err) => {
        console.error('TailFile had an error!', err);
    })
        .on('error', (err) => {
        console.error('A TailFile stream error was likely encountered', err);
    })
        .start()
        .catch((err) => {
        console.error('Cannot start.  Does the file exist?', err);
    });
    return newTail;
}
function getLatestFilePath() {
    let latestTime = 0;
    let latestFilePath = '';
    for (let i = 0; i < filesList.length; i++) {
        if (latestTime < filesList[i]['details']['mtimeMs']) {
            latestTime = filesList[i]['details']['mtimeMs'];
            latestFilePath = filesList[i]['path'];
        }
    }
    return latestFilePath;
}
function initializeOBSLogReader(obsPath, win) {
    fs.mkdirSync(obsPath, { recursive: true });
    window = win;
    console.log('chokidar.watch(obsPath)', obsPath);
    chokidar.watch(obsPath, { usePolling: true, }).on('add', (path, details) => {
        filesList.push({ path: path, details: details });
        if (tail) {
            tail.quit();
        }
        let filePath = getLatestFilePath();
        console.log('latest file', filePath);
        //doFullRead = true;
        if (doFullRead) {
            const data = fs.readFileSync(filePath, 'utf8');
            let events = parseChunk(data);
            handleEvents(events);
        }
        tail = startTail(filePath);
    });
}
exports.initializeOBSLogReader = initializeOBSLogReader;
function doFullReads() {
    doFullRead = true;
}
exports.doFullReads = doFullReads;
//# sourceMappingURL=obsLogReader.js.map
import * as chokidar from 'chokidar';
import * as fs from 'fs';
import TailFile from '@logdna/tail-file';

const sound = require("sound-play");

import uploadNft from './uploadNft';

let tail: any;
let filesList: any[] = [];
let doFullRead = false;
let window:any;


export function handleNewFile(filePath: string) {    
    console.log('new file: ', filePath)
    sound.play('assets/save.mp3')
    window.webContents.send('new-video', {
        filePath: filePath
    })
    uploadNft(filePath)
}

function parseChunk(chunk: string) {
    let lines = chunk.split('\n');
    let splitLine: string[];
    let timeStamp;
    let message;
    let events = [];
    for (let line of lines) {
        if (line.search('[Debug]')) {
            splitLine = line.split('[Debug]')
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message })
        }
        else if (line.search('[Info]')) {
            splitLine = line.split('[Info]');
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message })
        }
        else if (line.search('[Warning]')) {
            splitLine = line.split('[Warning]')
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message })
        }
        else if (line.search('[Warning]')) {
            splitLine = line.split('[Warning]')
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message })
        }
        else if (line.search('[Error]')) {
            splitLine = line.split('[Error]')
            timeStamp = splitLine[0];
            message = splitLine[0];
            events.push({ time: timeStamp, message: message })
        }
        else if (line.search('[]')) {}
        if (events[events.length-1]['message'].indexOf("Wrote replay buffer to ") !== -1) {
        }
    }
    return events;
}


function fireEvent(eventName, data) {
    if (eventName === 'newFile') {
	console.log('new file')
        handleNewFile(data);
    }
}

function handleEvents(events: any[]) {
    for (let event of events) {
	if (event['message'].indexOf("Wrote replay buffer to ") !== -1) {
	    let fileName = event['message'].split("Wrote replay buffer to ")[1];
            // fileName = fileName.slice(1, -2);
            fileName = fileName.slice(1, -1);
            fireEvent('newFile', fileName);
        }
    }
}

function startTail(filePath: string) {
    const newTail = new TailFile(filePath, { encoding: 'utf8' });
    newTail.on('data', (chunk) => {
        let events = parseChunk(chunk);
        handleEvents(events)
    })
        .on('tail_error', (err) => {
            console.error('TailFile had an error!', err)
        })
        .on('error', (err) => {
            console.error('A TailFile stream error was likely encountered', err)
        })
        .start()
        .catch((err) => {
            console.error('Cannot start.  Does the file exist?', err)
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


export function initializeOBSLogReader(
    obsPath: string,win: any) {
    fs.mkdirSync(obsPath, { recursive: true });
    window = win;
    console.log('chokidar.watch(obsPath)', obsPath);
    chokidar.watch(obsPath, { usePolling: true, }).on('add', (path, details) => {
        filesList.push({ path: path, details: details });
        if (tail) {
            tail.quit()
        }
        let filePath = getLatestFilePath()
        console.log('latest file', filePath);
        //doFullRead = true;
        if (doFullRead) {
            const data = fs.readFileSync(filePath, 'utf8');
            let events = parseChunk(data);
            handleEvents(events);
        }
        tail = startTail(filePath)
    });
}

export function doFullReads() {
    doFullRead = true;
}

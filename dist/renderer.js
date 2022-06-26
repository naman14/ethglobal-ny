"use strict";
// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process unless
// nodeIntegration is set to true in webPreferences.
// Use preload.js to selectively enable features
// needed in the renderer process.
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getStreamStatus = exports.createStream = void 0;
const electron_1 = require("electron");
const LIVEPEER_API_KEY = "96723baa-ee6f-4c6b-869b-0a110f8e27a6";
const database_1 = require("firebase/database");
const app_1 = require("firebase/app");
const database_2 = require("firebase/database");
const userName = 'naman';
electron_1.ipcRenderer.on('new-video', function (event, arg) {
    console.log('new video event', event, arg);
    document.getElementById('new-video').innerHTML = "New clip: " + arg.filePath;
});
let db;
initialiseDB();
checkStream();
function checkStream() {
    console.log('checking active streams');
    (0, database_1.get)((0, database_1.ref)(db, 'activeStreams/' + userName)).then((snapshot) => {
        if (snapshot.exists()) {
            console.log(snapshot.val());
            let streamId = snapshot.val();
            console.log(streamId);
            console.log('checking stream status');
            (0, exports.getStreamStatus)(LIVEPEER_API_KEY, streamId).then((response) => {
                console.log(response.data);
                let data = response.data;
                document.getElementById('stream-key-details-container').style.visibility = 'visible';
                document.getElementById('stream-key').innerHTML = data.streamKey;
                document.getElementById('stream-button').style.visibility = 'hidden';
                //   if (data.isActive) {
                //   } else {
                //     document.getElementById('stream-key-details-container').style.visibility =  'hidden'
                //   }
            });
        }
        else {
            console.log("No data available");
        }
    }).catch((error) => {
        console.error(error);
    });
}
function startStream() {
    console.log('starting stream');
    (0, exports.createStream)(LIVEPEER_API_KEY, "Naman Switch stream").then((response) => {
        console.log(response.data);
        let data = response.data;
        let newRef = (0, database_1.child)((0, database_1.ref)(db, 'streams/' + userName), data.id);
        (0, database_1.set)(newRef, {
            creator: userName,
            active: true,
            streamId: data.id,
            streamKey: data.streamKey,
            playbackId: data.playbackId,
            createdAt: data.createdAt,
            title: document.getElementById('sname').value,
            price: document.getElementById('sprice').value,
            freefor: document.getElementById('freefor').value,
            freefornft: document.getElementById('freefornft').value,
            mintPrice: document.getElementById('mprice').value,
            paymentAddress: document.getElementById('waddress').value
        });
        (0, database_1.set)((0, database_1.ref)(db, 'activeStreams/' + userName), data.id);
        document.getElementById('stream-key-details-container').style.visibility = 'visible';
        document.getElementById('stream-key').innerHTML = data.streamKey;
    });
}
const axios_1 = __importDefault(require("axios"));
const apiInstance = axios_1.default.create({
    baseURL: "https://livepeer.com/api/",
    timeout: 10000,
});
const createStream = (apiKey, name) => {
    return apiInstance.post("/stream", {
        name: name,
        profiles: [
            {
                name: "720p",
                bitrate: 2000000,
                fps: 30,
                width: 1280,
                height: 720,
            },
            {
                name: "480p",
                bitrate: 1000000,
                fps: 30,
                width: 854,
                height: 480,
            },
            {
                name: "360p",
                bitrate: 500000,
                fps: 30,
                width: 640,
                height: 360,
            },
        ],
    }, {
        headers: {
            "content-type": "application/json",
            authorization: `Bearer ${apiKey}`,
        },
    });
};
exports.createStream = createStream;
const getStreamStatus = (apiKey, streamId) => {
    return apiInstance.get(`/stream/${streamId}`, {
        headers: {
            "content-type": "application/json",
            authorization: `Bearer ${apiKey}`,
        },
    });
};
exports.getStreamStatus = getStreamStatus;
function initialiseDB() {
    // Your web app's Firebase configuration
    const firebaseConfig = {
        apiKey: "AIzaSyCLRQgLv7od_rHgXZNfWm4UQ7BFdsPHwvE",
        authDomain: "switch-ethglobal.firebaseapp.com",
        projectId: "switch-ethglobal",
        storageBucket: "switch-ethglobal.appspot.com",
        messagingSenderId: "555921925609",
        appId: "1:555921925609:web:b39a9fe5de5d3912a21778",
        databaseURL: "https://switch-ethglobal-default-rtdb.firebaseio.com"
    };
    // Initialize Firebase
    const app = (0, app_1.initializeApp)(firebaseConfig);
    db = (0, database_2.getDatabase)(app);
}
//# sourceMappingURL=renderer.js.map
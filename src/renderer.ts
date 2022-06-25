// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process unless
// nodeIntegration is set to true in webPreferences.
// Use preload.js to selectively enable features
// needed in the renderer process.

import { ipcRenderer } from 'electron'


ipcRenderer.on('new-video', function (event, arg) {
    console.log('new video event', event, arg)
    document.getElementById('new-video').innerHTML = "New clip: " + arg.filePath
})

// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";

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
const app = initializeApp(firebaseConfig);

import { getDatabase, ref, set } from "firebase/database";

const db = getDatabase(app);

set(ref(db, 'users/' + userId), {
    username: name,
    email: email,
    profile_picture : imageUrl
  });

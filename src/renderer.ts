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
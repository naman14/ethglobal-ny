// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process because
// `nodeIntegration` is turned off. Use `preload.js` to
// selectively enable features needed in the rendering
// process.

const {ipcRenderer} = require('electron')

const LIVEPEER_API_KEY = "96723baa-ee6f-4c6b-869b-0a110f8e27a6"

ipcRenderer.on('new-video', function (event, arg) {
    console.log('new video event', event, arg)
    document.getElementById('new-video').innerHTML = arg.filePath
})
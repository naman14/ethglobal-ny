// This file is required by the index.html file and will
// be executed in the renderer process for that window.
// No Node.js APIs are available in this process unless
// nodeIntegration is set to true in webPreferences.
// Use preload.js to selectively enable features
// needed in the renderer process.

import { ipcRenderer } from 'electron'

const LIVEPEER_API_KEY = "96723baa-ee6f-4c6b-869b-0a110f8e27a6"

import { ref, set, push, get, child } from "firebase/database";

import { initializeApp } from "firebase/app";
import { getDatabase } from "firebase/database";

const userName = 'naman'

ipcRenderer.on('new-video', function (event, arg) {
    console.log('new video event', event, arg)
    document.getElementById('new-video').innerHTML = "New clip: " + arg.filePath
})

let db: any

initialiseDB()
checkStream()

function checkStream() {
    console.log('checking active streams')
    get(ref(db, 'activeStreams/' + userName)).then((snapshot) => {
        if (snapshot.exists()) {
          console.log(snapshot.val());
        
          let streamId = snapshot.val()
          console.log(streamId)
        
          console.log('checking stream status')
          getStreamStatus(LIVEPEER_API_KEY, streamId).then((response) => {
            console.log(response.data)

              let data = response.data

              document.getElementById('stream-key-details-container').style.visibility =  'visible'
              document.getElementById('stream-key').innerHTML = data.streamKey

            //   if (data.isActive) {
              
            //   } else {
            //     document.getElementById('stream-key-details-container').style.visibility =  'hidden'
            //   }
          })
          

        } else {
          console.log("No data available");
        }
      }).catch((error) => {
        console.error(error);
      });
}

function startStream() {
    console.log('starting stream')
    document.getElementById('stream-button').innerHTML = 'Starting stream...'
    createStream(LIVEPEER_API_KEY, "Naman Switch stream").then((response) => {
        console.log(response.data)

        let data = response.data
        

        let newRef = child(ref(db, 'streams/' + userName), data.id)

        set(newRef, {
            creator: userName,
            active: true,
            streamId: data.id,
            streamKey: data.streamKey,
            playbackId: data.playbackId,
            createdAt: data.createdAt,
            title: 'Stream by ' + (document.getElementById('waddress') as HTMLInputElement)!.value,
            price: (document.getElementById('sprice') as HTMLInputElement)!.value,
            freefor: (document.getElementById('freefor') as HTMLInputElement)!.value,
            freefornft: (document.getElementById('freefornft') as HTMLInputElement)!.value,
            mintPrice: (document.getElementById('mprice') as HTMLInputElement)!.value,
            paymentAddress: (document.getElementById('waddress') as HTMLInputElement)!.value
          });

        set(ref(db, 'activeStreams/' + userName), data.id)

        document.getElementById('stream-key-details-container').style.visibility = 'visible'
        document.getElementById('stream-key').innerHTML = data.streamKey
        document.getElementById('stream-button').innerHTML = 'Stream started'

    })
}

import axios from "axios";

const apiInstance = axios.create({
  baseURL: "https://livepeer.com/api/",
  timeout: 10000,
});

export const createStream = (apiKey: string, name: string): Promise<any> => {
  return apiInstance.post(
    "/stream",
    {
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
    },
    {
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
    }
  );
};

export const getStreamStatus = (
  apiKey: string,
  streamId: string
): Promise<any> => {
  return apiInstance.get(`/stream/${streamId}`, {
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${apiKey}`,
    },
  });
};

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
    const app = initializeApp(firebaseConfig);

    db = getDatabase(app);
}
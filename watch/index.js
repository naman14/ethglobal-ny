import { initializeApp, } from "https://www.gstatic.com/firebasejs/9.8.4/firebase-app.js";

import { get, ref, getDatabase } from "https://cdnjs.cloudflare.com/ajax/libs/firebase/9.8.4/firebase-database.min.js"

const LIVEPEER_API_KEY = "96723baa-ee6f-4c6b-869b-0a110f8e27a6"

const urlParams = new URLSearchParams(window.location.search);
const username = urlParams.get('u');

console.log('user: ' + username)

let db;

initialiseDB()
fetchStream(username)

function initialiseDB() {
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
      db = getDatabase(app)
      console.log('initialised')

}

function fetchStream(username) {

    console.log(db)
    console.log('checking active streams')

    get(ref(db, 'activeStreams/' + username)).then((snapshot) => {
        console.log(snapshot)
        if (snapshot.exists()) {
          console.log(snapshot.val());
        
          let streamId = snapshot.val()
          console.log(streamId)
        
          console.log('checking stream status')

          get(ref(db, 'activeStreams/' + username + '/' + streamId)).then((snapshot) => {

            let streamInfo = snapshot.val()
            
            document.getElementById('freefor-text').innerHTML = streamInfo.freefor + ' minutes'
            document.getElementById('sprice-text').innerHTML = streamInfo.price + ' Ξ/min'
            document.getElementById('nft-count-text').innerHTML = '1'

            return

            getStreamStatus(LIVEPEER_API_KEY, streamId).then((response) => {
                console.log(response.data)
    
                  let data = response.data

                  var video = videojs("video");

                    video.src({
                        type: 'video/mp4',
                        src: 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'
                      });

                    // source.setAttribute('src', `https://livepeercdn.com/hls/${data.playbackId}/index.m3u8`);
                    // source.setAttribute('src', `http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8`);

                    // source.setAttribute('type', 'application/x-mpegURL');

                    // video.appendChild(source);

                // document.getElementById('stream-title').innerHTML = streamInfo.title
              })

          }).catch((error) => {
            console.error(error);
        });

       
          

        } else {
          console.log("No data available");
        }
      }).catch((error) => {
        console.error(error);
      });
}

import "https://cdnjs.cloudflare.com/ajax/libs/axios/1.0.0-alpha.1/axios.min.js";

const apiInstance = axios.create({
  baseURL: "https://livepeer.com/api/",
  timeout: 10000,
});

export const getStreamStatus = (
    apiKey,
    streamId
  ) => {
    return apiInstance.get(`/stream/${streamId}`, {
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
    });
  };
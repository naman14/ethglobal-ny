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

import { getDatabase } from "firebase/database";

const db = getDatabase(app);

export default db
import { initializeApp, } from "https://www.gstatic.com/firebasejs/9.8.4/firebase-app.js";

import { get, ref, getDatabase, onValue } from "https://cdnjs.cloudflare.com/ajax/libs/firebase/9.8.4/firebase-database.min.js"

import * as superfluid from 'https://cdn.skypack.dev/pin/@superfluid-finance/sdk-core@v0.4.2-IVr6JAGzcFoFBoUZkPy5/mode=imports/optimized/@superfluid-finance/sdk-core.js'

import { ethers } from "https://cdn.ethers.io/lib/ethers-5.2.esm.min.js";

const LIVEPEER_API_KEY = "96723baa-ee6f-4c6b-869b-0a110f8e27a6"

const urlParams = new URLSearchParams(window.location.search);
const username = urlParams.get('u');

console.log('user: ' + username)

let db;
let currentNft;

// initialiseDB()
// fetchStream(username)
// watchNftDrops()
// watchChat()

setupSuperfluid()

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

async function setupSuperfluid() {
    const config = {
        hostAddress: "0x3E14dC1b13c488a8d5D310918780c983bD5982E7",
        cfaV1Address: "0x6EeE6060f715257b970700bc2656De21dEdF074C",
        idaV1Address: "0xB0aABBA4B2783A72C52956CDEF62d438ecA2d7a1"
      };
      
    const provider = new ethers.providers.Web3Provider(window.ethereum);
    const sf = await superfluid.Framework.create({
        networkName: "polygon-mumbai",
        provider: provider
      });

    const signer = sf.createSigner({ web3Provider: provider });

    const paymentAddress = "0xCE9F8B4E91582bFF8cD4C2eB0C811e918779715a"
    const supertokenAddress = "0x96b82b65acf7072efeb00502f45757f254c2a0d4"
    const cfa = sf.cfaV1
    console.log(cfa)

    const maticx = await sf.loadSuperToken(supertokenAddress)

    const approveOp = maticx.approve({ receiver: paymentAddress, amount: "10000" });

    const createFlowOperation = sf.cfaV1.createFlow({
        sender: "0x16b1025cD1A83141bf93E47dBC316f34f27f2e76",
        receiver: "0xCE9F8B4E91582bFF8cD4C2eB0C811e918779715a",
        superToken: "0x96b82b65acf7072efeb00502f45757f254c2a0d4",
        flowRate: "1000000000"
      });
    const batchCall = sf.batchCall([approveOp, createFlowOperation]);
    const txn = await batchCall.exec(signer);
    const txnReceipt = await txn.wait();
    console.log(txnReceipt)
}

    

function fetchStream(username) {

    console.log(db)
    console.log('checking active streams')

    return
    get(ref(db, 'activeStreams/' + username)).then((snapshot) => {
        console.log(snapshot)
        if (snapshot.exists()) {
          console.log(snapshot.val());
        
          let streamId = snapshot.val()
          console.log(streamId)
        
          console.log('checking stream status')

          get(ref(db, 'streams/' + username + '/' + streamId)).then((snapshot) => {

            let streamInfo = snapshot.val()
            
            document.getElementById('freefor-text').innerHTML = streamInfo.freefor + ' minutes'
            document.getElementById('sprice-text').innerHTML = streamInfo.price + ' Ξ/min'
            document.getElementById('nft-count-text').innerHTML = '1'

            getStreamStatus(LIVEPEER_API_KEY, streamId).then((response) => {
                console.log(response.data)
    
                  let data = response.data

                  var video = videojs("video");
                
                  const playbackUrl = `https://livepeercdn.com/hls/${data.playbackId}/index.m3u8`
                  console.log('playback url: ' + playbackUrl)

                    video.src({
                        type: 'application/x-mpegURL',
                        src: playbackUrl
                      });

                document.getElementById('stream-title').innerHTML = streamInfo.title
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

function watchNftDrops() {
    const nftRef = ref(db, 'nfts/' + username);
    onValue(nftRef, (snapshot) => {

      const data = snapshot.val()
        
      if (!data) return

      console.log(data)
      
      let found = false
      Object.values(data).forEach((nft) => {
        if (!found) {
            found = true
            currentNft = nft
        }
      })
      
      let minted = currentNft.minted

      if (!minted) {
        document.getElementById('mint-nft-button').innerHTML = 'Mint now'
        document.getElementById('mint-nft-button').style.backgroundColor = '#4A7DFF'

      } else {
        document.getElementById('mint-nft-button').innerHTML = 'Minted by ' + currentNft.mintedBy
        document.getElementById('mint-nft-button').style.backgroundColor = '#0E0F18'
      }
    });
}

function watchChat() {
    const nftRef = ref(db, 'chats/' + username);
    onValue(nftRef, (snapshot) => {

      const data = snapshot.val()
        
      if (!data) return

      console.log(data)
      let text = ''

      Object.values(data).forEach((message) => {
        text = text + '\n\n' + message
      })
      
      document.getElementById('chat-messages').innerHTML = text
    });
}

document.getElementById('mint-nft-button').addEventListener('click', function() {
    
    mintNft()
 }, false);


async function mintNft() {
    if (currentNft && currentNft.minted) {
        window.open(currentNft.openseaUrl, '_blank').focus();
    } else {
        console.log('minting nft')
        document.getElementById('mint-nft-button').innerHTML = 'Minting...'

        const apiOpts = {
            auth: { apiKey: LIVEPEER_API_KEY },
            // defaults to current origin if not specified
            endpoint: videonft.api.prodApiEndpoint
          };
        const minter = videonft.minter
        const uploader = new minter.Uploader();
        const sdk = new minter.Api(apiOpts);

        const web3 = new videonft.minter.FullMinter({}, { ethereum, chainId: 80001 }).web3;

        console.log(web3)

        let ipfsUrl = currentNft.tokenUri

        const tx = await web3.mintNft(ipfsUrl);
        const nftInfo = await web3.getMintedNftInfo(tx);
        console.log(`minted NFT on contract ${nftInfo.contractAddress} with ID ${nftInfo.tokenId}`);

        set(ref(db, 'nfts/' + username), {
            creator: currentNft.creator,
            minted: true,
            tokenUri: ipfs.nftMetadataUrl,
            nftInfo: nftInfo
        });

        let newChatRef = push(ref(db, 'chats/' + username))
        set(newChatRef, 'NFT minted! '+ nftInfo.opensea.tokenUrl);
    }
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
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const video_nft_1 = require("@livepeer/video-nft");
const firebase_1 = __importDefault(require("./firebase"));
const database_1 = require("firebase/database");
const LIVEPEER_API_KEY = "96723baa-ee6f-4c6b-869b-0a110f8e27a6";
const apiOpts = {
    auth: { apiKey: LIVEPEER_API_KEY },
    // defaults to current origin if not specified
    endpoint: video_nft_1.videonft.api.prodApiEndpoint
};
async function uploadVideo(filePath) {
    const nftTitle = "Naman Switch #1";
    const minter = video_nft_1.videonft.minter;
    const uploader = new minter.Uploader();
    const sdk = new minter.Api(apiOpts);
    let asset = await uploader.useFile(filePath, file => {
        console.log('uploading file');
        return sdk.createAsset(nftTitle, file, printProgress);
    });
    const nftMetadata = {
        description: nftTitle,
        traits: { 'creator': 'Naman', 'id': '1' }
    };
    let ipfs = await sdk.exportToIPFS(asset.id, nftMetadata, printProgress);
    console.log(`Export successful! Result: \n${JSON.stringify(ipfs, null, 2)}`);
    const mintUrl = 'https://livepeer.studio/mint-nft?tokenUri=' + ipfs.nftMetadataUrl;
    console.log('mint nft at: ' + mintUrl);
    const userName = 'naman';
    let newRef = (0, database_1.push)((0, database_1.ref)(firebase_1.default, 'nfts/' + userName));
    (0, database_1.set)(newRef, {
        creator: userName,
        minted: false,
        tokenUri: ipfs.nftMetadataUrl,
        mintedBy: null
    });
    let newChatRef = (0, database_1.push)((0, database_1.ref)(firebase_1.default, 'chats/' + userName));
    (0, database_1.set)(newChatRef, 'NFT dropped!. Mint now at: ' + mintUrl);
}
exports.default = uploadVideo;
function printProgress(progress) {
    console.log(` - progress: ${100 * progress}%`);
}
//# sourceMappingURL=uploadNft.js.map
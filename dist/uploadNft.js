"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const video_nft_1 = require("@livepeer/video-nft");
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
    console.log('mint nft at: ' + 'https://livepeer.studio/mint-nft?tokenUri=' + ipfs.nftMetadataUrl);
}
exports.default = uploadVideo;
function printProgress(progress) {
    console.log(` - progress: ${100 * progress}%`);
}
//# sourceMappingURL=uploadNft.js.map
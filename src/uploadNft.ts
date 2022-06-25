import { videonft } from '@livepeer/video-nft'

import db from './firebase'

import { getDatabase, ref, set, push } from "firebase/database";

const LIVEPEER_API_KEY = "96723baa-ee6f-4c6b-869b-0a110f8e27a6"

const apiOpts = {
    auth: { apiKey: LIVEPEER_API_KEY },
    // defaults to current origin if not specified
    endpoint: videonft.api.prodApiEndpoint
  };

export default async function uploadVideo(filePath: string) {
    const nftTitle = "Naman Switch #1"

    const minter = videonft.minter
    const uploader = new minter.Uploader();
	const sdk = new minter.Api(apiOpts);

    let asset = await uploader.useFile(filePath, file => {
		console.log('uploading file')
		return sdk.createAsset(nftTitle, file, printProgress);
	});

    const nftMetadata = {
        description: nftTitle,
        traits: { 'creator': 'Naman', 'id': '1' }
      };

    let ipfs = await sdk.exportToIPFS(
		asset.id,
		nftMetadata,
		printProgress
	);

	console.log(
		`Export successful! Result: \n${JSON.stringify(ipfs, null, 2)}`
	);

    const mintUrl = 'https://livepeer.studio/mint-nft?tokenUri=' + ipfs.nftMetadataUrl
    console.log('mint nft at: ' + mintUrl)

    const userName = 'naman'

    let newRef = push(ref(db, 'nfts/' + userName))

    set(newRef, {
        creator: userName,
        minted: false,
        tokenUri: ipfs.nftMetadataUrl,
        mintedBy: null
      });

    let newChatRef = push(ref(db, 'chats/' + userName))
    set(newChatRef, 'NFT dropped!. Mint now at: ' + mintUrl);

}

function printProgress(progress: number) {
	console.log(` - progress: ${100 * progress}%`);
}
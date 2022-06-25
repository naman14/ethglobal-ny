"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const axios_1 = __importDefault(require("axios"));
/**
 * calls the /stream route of Livepeer.com APIs to create a new stream.
 * The response returns the playbackId and streamKey.
 * With this data available the ingest and playback urls would respectively be:
 * Ingest URL: rtmp://rtmp.livepeer.com/live/{stream-key}
 * Playback URL: https://cdn.livepeer.com/hls/{playbackId}/index.m3u8
 */
exports.default = async (req, res) => {
    if (req.method === "POST") {
        const authorizationHeader = req.headers && req.headers["authorization"];
        const streamName = req.body && req.body.name;
        const streamProfiles = req.body && req.body.profiles;
        try {
            const createStreamResponse = await axios_1.default.post("https://livepeer.com/api/stream", {
                name: streamName,
                profiles: streamProfiles,
            }, {
                headers: {
                    "content-type": "application/json",
                    authorization: authorizationHeader, // API Key needs to be passed as a header
                },
            });
            if (createStreamResponse && createStreamResponse.data) {
                res.statusCode = 200;
                res.json({ ...createStreamResponse.data });
            }
            else {
                res.statusCode = 500;
                res.json({ error: "Something went wrong" });
            }
        }
        catch (error) {
            res.statusCode = 500;
            // Handles Invalid API key error
            if (error.response.status === 403) {
                res.statusCode = 403;
            }
            res.json({ error });
        }
    }
};
//# sourceMappingURL=createStream.js.map
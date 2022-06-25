"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getStreamStatus = exports.createStream = void 0;
const axios_1 = __importDefault(require("axios"));
const apiInstance = axios_1.default.create({
    baseURL: "/api/",
    timeout: 10000,
});
const createStream = (apiKey, name) => {
    return apiInstance.post("/stream", {
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
    }, {
        headers: {
            "content-type": "application/json",
            authorization: `Bearer ${apiKey}`,
        },
    });
};
exports.createStream = createStream;
const getStreamStatus = (apiKey, streamId) => {
    return apiInstance.get(`/stream/${streamId}`, {
        headers: {
            "content-type": "application/json",
            authorization: `Bearer ${apiKey}`,
        },
    });
};
exports.getStreamStatus = getStreamStatus;
//# sourceMappingURL=apiFactory.js.map
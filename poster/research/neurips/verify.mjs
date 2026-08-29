#!/usr/bin/env node

import {createHash} from "node:crypto";
import {readFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const sources = JSON.parse(await readFile(path.join(here, "sources.json"), "utf8"));
const manifest = JSON.parse(await readFile(path.join(here, "manifest.generated.json"), "utf8"));
const notes = await readFile(path.join(here, "notes.md"), "utf8");
const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const crcTable = new Uint32Array(256);

for (let value = 0; value < crcTable.length; value += 1) {
  let crc = value;
  for (let bit = 0; bit < 8; bit += 1) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  crcTable[value] = crc >>> 0;
}

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function inspectPng(bytes, label) {
  if (!bytes.subarray(0, pngSignature.length).equals(pngSignature)) throw new Error(`${label}: invalid PNG signature`);

  let offset = pngSignature.length;
  let chunkIndex = 0;
  let sawIhdr = false;
  let sawPlte = false;
  let sawIdat = false;
  let endedIdat = false;
  let sawIend = false;
  let width;
  let height;
  let colorType;

  while (offset < bytes.length) {
    if (bytes.length - offset < 12) throw new Error(`${label}: truncated PNG chunk header`);
    const length = bytes.readUInt32BE(offset);
    const typeStart = offset + 4;
    const dataStart = typeStart + 4;
    const dataEnd = dataStart + length;
    const chunkEnd = dataEnd + 4;
    if (chunkEnd > bytes.length) throw new Error(`${label}: truncated PNG chunk data`);

    const typeBytes = bytes.subarray(typeStart, dataStart);
    const type = typeBytes.toString("ascii");
    if (!/^[A-Za-z]{4}$/.test(type)) throw new Error(`${label}: invalid PNG chunk type`);
    const expectedCrc = bytes.readUInt32BE(dataEnd);
    const measuredCrc = crc32(bytes.subarray(typeStart, dataEnd));
    if (measuredCrc !== expectedCrc) throw new Error(`${label}: ${type} chunk CRC mismatch`);
    if (chunkIndex === 0 && type !== "IHDR") throw new Error(`${label}: IHDR is not the first PNG chunk`);
    if ((typeBytes[0] & 0x20) === 0 && !["IHDR", "PLTE", "IDAT", "IEND"].includes(type)) {
      throw new Error(`${label}: unknown critical PNG chunk ${type}`);
    }

    if (type === "IHDR") {
      if (sawIhdr || length !== 13) throw new Error(`${label}: invalid IHDR chunk`);
      sawIhdr = true;
      width = bytes.readUInt32BE(dataStart);
      height = bytes.readUInt32BE(dataStart + 4);
      const bitDepth = bytes[dataStart + 8];
      colorType = bytes[dataStart + 9];
      const validDepths = new Map([
        [0, [1, 2, 4, 8, 16]],
        [2, [8, 16]],
        [3, [1, 2, 4, 8]],
        [4, [8, 16]],
        [6, [8, 16]],
      ]);
      if (!width || !height || !validDepths.get(colorType)?.includes(bitDepth)) throw new Error(`${label}: invalid IHDR image fields`);
      if (bytes[dataStart + 10] !== 0 || bytes[dataStart + 11] !== 0 || ![0, 1].includes(bytes[dataStart + 12])) {
        throw new Error(`${label}: unsupported IHDR compression, filter, or interlace value`);
      }
    } else if (type === "PLTE") {
      if (!sawIhdr || sawPlte || sawIdat || length === 0 || length % 3 !== 0 || length > 768) throw new Error(`${label}: invalid PLTE chunk`);
      if ([0, 4].includes(colorType)) throw new Error(`${label}: PLTE is forbidden for this color type`);
      sawPlte = true;
    } else if (type === "IDAT") {
      if (!sawIhdr || endedIdat || sawIend) throw new Error(`${label}: invalid IDAT chunk order`);
      if (colorType === 3 && !sawPlte) throw new Error(`${label}: indexed-color PNG has no PLTE chunk`);
      sawIdat = true;
    } else if (type === "IEND") {
      if (!sawIhdr || !sawIdat || sawIend || length !== 0) throw new Error(`${label}: invalid IEND chunk`);
      sawIend = true;
      if (chunkEnd !== bytes.length) throw new Error(`${label}: bytes found after IEND`);
    } else if (sawIdat) {
      endedIdat = true;
    }

    offset = chunkEnd;
    chunkIndex += 1;
  }

  if (!sawIhdr || !sawIdat || !sawIend) throw new Error(`${label}: incomplete PNG chunk structure`);
  return {width, height};
}

async function fetchRequired(url, label) {
  const response = await fetch(url, {
    headers: {"user-agent": "CEL-poster-source-verifier/1.0"},
    redirect: "follow",
    signal: AbortSignal.timeout(30_000),
  });
  if (response.status !== 200) throw new Error(`${label}: expected HTTP 200, received ${response.status}`);
  return response;
}

if (sources.sources.length !== 5 || manifest.assets.length !== 5) throw new Error("Expected exactly five real source and manifest rows");
const sourcesById = new Map();
const sourceUrls = new Set();
for (const source of sources.sources) {
  if (!idPattern.test(source.id) || sourcesById.has(source.id)) throw new Error(`Invalid or duplicate source ID: ${source.id}`);
  if (!/^https:\/\/neurips\.cc\/virtual\/\d{4}\/poster\/\d+$/.test(source.pageUrl)) throw new Error(`${source.id}: invalid official event page`);
  if (!source.assetUrl.startsWith("https://neurips.cc/media/") || source.observedFormat !== "PNG") throw new Error(`${source.id}: invalid official asset record`);
  if (sourceUrls.has(source.assetUrl)) throw new Error(`${source.id}: duplicate official asset URL`);
  sourcesById.set(source.id, source);
  sourceUrls.add(source.assetUrl);
}

const hashes = new Set();
const manifestIds = new Set();
for (const asset of manifest.assets) {
  const source = sourcesById.get(asset.id);
  if (!source) throw new Error(`Padded or unknown manifest row: ${asset.id}`);
  if (manifestIds.has(asset.id)) throw new Error(`Duplicate manifest row: ${asset.id}`);
  manifestIds.add(asset.id);
  if (asset.requestedUrl !== source.assetUrl || !asset.finalUrl.startsWith("https://neurips.cc/media/")) throw new Error(`${asset.id}: URL provenance mismatch`);
  if (asset.filename !== `${asset.id}.png` || asset.observedFormat !== "PNG" || asset.mediaType !== "image/png") throw new Error(`${asset.id}: format mismatch`);
  const pageResponse = await fetchRequired(source.pageUrl, `${asset.id} official page`);
  const pageContentType = pageResponse.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase();
  if (pageContentType !== "text/html") throw new Error(`${asset.id}: official page is not HTML`);
  const pageHtml = await pageResponse.text();
  const assetPath = new URL(source.assetUrl).pathname;
  const assetBasename = path.posix.basename(assetPath);
  if (![source.assetUrl, assetPath, assetBasename].some((reference) => pageHtml.includes(reference))) {
    throw new Error(`${asset.id}: official page does not name its exact poster asset URL, path, or basename`);
  }

  const assetResponse = await fetchRequired(source.assetUrl, `${asset.id} official asset`);
  const remoteMediaType = assetResponse.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase();
  if (remoteMediaType !== "image/png") throw new Error(`${asset.id}: official asset media type is ${remoteMediaType ?? "missing"}, not image/png`);
  if (assetResponse.url !== asset.finalUrl) throw new Error(`${asset.id}: fetched final URL does not match manifest`);

  const localBytes = await readFile(path.join(here, "assets", asset.filename));
  const remoteBytes = Buffer.from(await assetResponse.arrayBuffer());
  if (localBytes.length < 10000 || remoteBytes.length < 10000) throw new Error(`${asset.id}: corrupt or placeholder PNG`);
  if (!remoteBytes.equals(localBytes)) throw new Error(`${asset.id}: local asset is not byte-for-byte equal to the official asset`);

  const localDimensions = inspectPng(localBytes, `${asset.id} local asset`);
  const remoteDimensions = inspectPng(remoteBytes, `${asset.id} official asset`);
  const localSha256 = createHash("sha256").update(localBytes).digest("hex");
  const remoteSha256 = createHash("sha256").update(remoteBytes).digest("hex");
  if (localSha256 !== asset.sha256 || remoteSha256 !== asset.sha256) throw new Error(`${asset.id}: fetched/local hash does not match manifest`);
  if (localBytes.length !== asset.byteSize || remoteBytes.length !== asset.byteSize) throw new Error(`${asset.id}: fetched/local size does not match manifest`);
  if (localDimensions.width !== asset.widthPx || localDimensions.height !== asset.heightPx ||
      remoteDimensions.width !== asset.widthPx || remoteDimensions.height !== asset.heightPx) {
    throw new Error(`${asset.id}: fetched/local dimensions do not match manifest`);
  }
  if (localDimensions.width < 1500 || localDimensions.height < 1000) throw new Error(`${asset.id}: dimensions are inconsistent with a real poster`);
  if (hashes.has(localSha256)) throw new Error(`${asset.id}: duplicate poster bytes`);
  hashes.add(localSha256);
}
if (hashes.size !== sourcesById.size || manifestIds.size !== sourcesById.size) throw new Error("Source/asset coverage mismatch");
const observationLines = notes.split("\n").filter((line) => line.startsWith("- **Observation"));
if (observationLines.length === 0) throw new Error("No direct visual observations found in notes.md");
for (const line of observationLines) {
  const id = line.match(/^\- \*\*Observation \[([^\]]+)\]/)?.[1];
  if (!id || !sourcesById.has(id)) throw new Error(`Observed claim has no valid source ID: ${line}`);
}
for (const id of sourcesById.keys()) {
  if (!observationLines.some((line) => line.includes(`[${id}]`))) throw new Error(`No observed claim cites ${id}`);
}
console.log(`Verified ${hashes.size} unique official NeurIPS PNG posters`);

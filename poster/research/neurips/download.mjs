#!/usr/bin/env node

import {createHash} from "node:crypto";
import {mkdir, readFile, writeFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const sources = JSON.parse(await readFile(path.join(here, "sources.json"), "utf8"));
const assetsDir = path.join(here, "assets");

function pngDimensions(bytes) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (bytes.length < 24 || !bytes.subarray(0, 8).equals(signature)) throw new Error("Asset is not a PNG");
  if (bytes.toString("ascii", 12, 16) !== "IHDR") throw new Error("PNG has no IHDR chunk");
  return {width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20)};
}

await mkdir(assetsDir, {recursive: true});
const manifest = [];
for (const source of sources.sources) {
  const response = await fetch(source.assetUrl, {redirect: "follow", headers: {"user-agent": "CEL-poster-research/1.0"}});
  if (!response.ok) throw new Error(`${source.id}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const dimensions = pngDimensions(bytes);
  const filename = `${source.id}.png`;
  await writeFile(path.join(assetsDir, filename), bytes);
  manifest.push({
    id: source.id,
    filename,
    requestedUrl: source.assetUrl,
    finalUrl: response.url,
    mediaType: response.headers.get("content-type")?.split(";")[0] ?? null,
    observedFormat: "PNG",
    widthPx: dimensions.width,
    heightPx: dimensions.height,
    byteSize: bytes.length,
    sha256: createHash("sha256").update(bytes).digest("hex")
  });
}

await writeFile(path.join(here, "manifest.generated.json"), `${JSON.stringify({schemaVersion: 1, assets: manifest}, null, 2)}\n`);
console.log(`Downloaded ${manifest.length} official poster assets`);
for (const asset of manifest) console.log(`${asset.id}: ${asset.widthPx}×${asset.heightPx}, ${asset.byteSize} bytes, ${asset.mediaType}`);

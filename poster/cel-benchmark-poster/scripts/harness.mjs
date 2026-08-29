import { createReadStream, existsSync } from 'node:fs'
import { stat } from 'node:fs/promises'
import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

export const projectDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
export const repositoryDir = path.resolve(projectDir, '../..')
export const distDir = path.join(projectDir, 'dist')
export const deliverablesDir = path.join(projectDir, 'deliverables')

const mimeTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.svg', 'image/svg+xml'],
  ['.png', 'image/png'],
])

function browserExecutable() {
  const candidates = [
    process.env.POSTER_CHROMIUM_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
  ].filter(Boolean)
  const executable = candidates.find((candidate) => existsSync(candidate))
  if (!executable) throw new Error('No local Chromium executable found; set POSTER_CHROMIUM_PATH')
  return executable
}

async function startStaticServer() {
  const server = http.createServer(async (request, response) => {
    try {
      const requestPath = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname)
      const relativePath = requestPath === '/' ? 'index.html' : requestPath.replace(/^\/+/, '')
      const absolutePath = path.resolve(distDir, relativePath)
      if (!(absolutePath === distDir || absolutePath.startsWith(`${distDir}${path.sep}`))) {
        response.writeHead(403).end('Forbidden')
        return
      }
      const fileStat = await stat(absolutePath)
      if (!fileStat.isFile()) throw new Error('Not a file')
      response.writeHead(200, {
        'Content-Type': mimeTypes.get(path.extname(absolutePath)) ?? 'application/octet-stream',
        'Cache-Control': 'no-store',
      })
      createReadStream(absolutePath).pipe(response)
    } catch {
      response.writeHead(404).end('Not found')
    }
  })
  await new Promise((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, '127.0.0.1', resolve)
  })
  const address = server.address()
  if (!address || typeof address === 'string') throw new Error('Could not resolve local render server')
  return { server, url: `http://127.0.0.1:${address.port}/` }
}

export async function withPosterPage(callback) {
  const { server, url } = await startStaticServer()
  const failures = []
  const browser = await chromium.launch({
    executablePath: browserExecutable(),
    headless: true,
    args: ['--disable-dev-shm-usage'],
  })
  const context = await browser.newContext({ viewport: { width: 1920, height: 1600 }, deviceScaleFactor: 1 })
  const page = await context.newPage()
  page.on('requestfailed', (request) => failures.push(`${request.url()}: ${request.failure()?.errorText ?? 'request failed'}`))
  page.on('response', (response) => {
    if (response.status() >= 400) failures.push(`${response.url()}: HTTP ${response.status()}`)
  })
  try {
    await page.goto(url, { waitUntil: 'networkidle' })
    await page.evaluate(() => document.fonts.ready)
    return await callback({ page, failures, url })
  } finally {
    await context.close()
    await browser.close()
    await new Promise((resolve) => server.close(resolve))
  }
}

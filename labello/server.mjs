import { createServer } from 'https'
import { parse } from 'url'
import next from 'next'
import fs from 'fs'
import path from 'path'

const dev = process.env.NODE_ENV !== 'production'
const hostname = 'localhost'
const port = 3000

// Prepare the Next.js app
const app = next({ dev, hostname, port })
const handle = app.getRequestHandler()

const httpsOptions = {
  key: fs.readFileSync(path.join(process.cwd(), '.ssl/key.pem')),
  cert: fs.readFileSync(path.join(process.cwd(), '.ssl/cert.pem')),
}

app.prepare().then(() => {
  createServer(httpsOptions, async (req, res) => {
    try {
      const parsedUrl = parse(req.url, true)
      await handle(req, res, parsedUrl)
    } catch (err) {
      console.error('Error occurred handling', req.url, err)
      res.statusCode = 500
      res.end('internal server error')
    }
  })
    .once('error', (err) => {
      console.error(err)
      process.exit(1)
    })
    .listen(port, () => {
      console.log(`> Ready on https://${hostname}:${port}`)
      console.log(`> Network: https://192.168.1.79:${port}`)
    })
})
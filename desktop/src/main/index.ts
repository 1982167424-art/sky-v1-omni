import { app, BrowserWindow, ipcMain, shell, dialog } from 'electron'
import { spawn, exec, ChildProcess } from 'child_process'
import { createServer } from 'net'
import * as path from 'path'
import * as fs from 'fs'
import { autoUpdater } from 'electron-updater'
import { IPty, spawn as ptySpawn } from 'node-pty'

let mainWindow: BrowserWindow | null = null
let backendProcess: ChildProcess | null = null
let backendPort: number = 8765
let backendRunning: boolean = false
let backendLogs: string[] = []
const MAX_LOG_LINES = 500

let ptyProcess: IPty | null = null

const repoRoot = path.resolve(__dirname, '..', '..', '..')

function appendLog(line: string) {
  backendLogs.push(line)
  if (backendLogs.length > MAX_LOG_LINES) {
    backendLogs = backendLogs.slice(-MAX_LOG_LINES)
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:log', line)
  }
}

function broadcastStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:status-change', {
      running: backendRunning,
      port: backendPort
    })
  }
}

function findAvailablePort(startPort: number): Promise<number> {
  return new Promise((resolve) => {
    const server = createServer()
    server.unref()
    server.on('error', () => {
      resolve(findAvailablePort(startPort + 1))
    })
    server.listen(startPort, '127.0.0.1', () => {
      const addr = server.address() as any
      const port = addr.port
      server.close(() => resolve(port))
    })
  })
}

function startBackend() {
  if (backendRunning) return

  const pythonEnv = { ...process.env }
  if (pythonEnv.PYTHONPATH) {
    pythonEnv.PYTHONPATH = repoRoot + path.delimiter + pythonEnv.PYTHONPATH
  } else {
    pythonEnv.PYTHONPATH = repoRoot
  }

  findAvailablePort(8765).then((port) => {
    backendPort = port
    appendLog(`[backend] Starting backend on port ${port}...`)

    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3'
    backendProcess = spawn(
      pythonCmd,
      ['-m', 'sky_v1.cli.main', 'serve', '--host', '127.0.0.1', '--port', String(port)],
      {
        cwd: repoRoot,
        env: pythonEnv
      }
    )

    backendRunning = true
    broadcastStatus()

    backendProcess.stdout?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n')
      lines.forEach((l) => l && appendLog(`[stdout] ${l}`))
    })

    backendProcess.stderr?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n')
      lines.forEach((l) => l && appendLog(`[stderr] ${l}`))
    })

    backendProcess.on('close', (code: number | null) => {
      appendLog(`[backend] Process exited with code ${code}`)
      backendRunning = false
      backendProcess = null
      broadcastStatus()
    })

    backendProcess.on('error', (err: Error) => {
      appendLog(`[backend] Error: ${err.message}`)
      backendRunning = false
      backendProcess = null
      broadcastStatus()
    })
  })
}

function stopBackend() {
  if (backendProcess && !backendProcess.killed) {
    appendLog('[backend] Stopping backend...')
    backendProcess.kill('SIGTERM')
    setTimeout(() => {
      if (backendProcess && !backendProcess.killed) {
        backendProcess.kill('SIGKILL')
      }
    }, 3000)
  }
}

function restartBackend() {
  stopBackend()
  setTimeout(() => startBackend(), 1000)
}

function setupIpc() {
  ipcMain.handle('backend:status', () => {
    return {
      running: backendRunning,
      port: backendPort,
      logs: backendLogs.slice(-100)
    }
  })

  ipcMain.handle('backend:restart', () => {
    restartBackend()
    return true
  })

  ipcMain.handle('backend:stop', () => {
    stopBackend()
    return true
  })

  ipcMain.handle('app:get-version', () => {
    return app.getVersion()
  })

  ipcMain.handle('app:open-external', (_event, url: string) => {
    return shell.openExternal(url)
  })

  ipcMain.handle('github:cli', async (_event, args: string[]) => {
    return new Promise((resolve) => {
      const cmd = `gh ${args.join(' ')}`
      exec(cmd, { maxBuffer: 10 * 1024 * 1024 }, (err, stdout, stderr) => {
        resolve({
          stdout: stdout || '',
          stderr: stderr || '',
          code: err?.code ?? 0
        })
      })
    })
  })

  ipcMain.handle('api:fetch', async (_event, method: string, urlPath: string, body?: any) => {
    try {
      const url = `http://127.0.0.1:${backendPort}/v1${urlPath}`
      const init: RequestInit = {
        method,
        headers: { 'Content-Type': 'application/json' }
      }
      if (body !== undefined && method !== 'GET') {
        init.body = JSON.stringify(body)
      }
      const resp = await fetch(url, init)
      const text = await resp.text()
      try {
        return { ok: resp.ok, status: resp.status, data: JSON.parse(text) }
      } catch {
        return { ok: resp.ok, status: resp.status, data: text }
      }
    } catch (e: any) {
      return { ok: false, status: 0, data: null, error: e.message }
    }
  })

  ipcMain.handle('rag:list-files', async () => {
    try {
      const presetsDir = path.join(repoRoot, 'sky_v1', 'rag', 'presets')
      if (!fs.existsSync(presetsDir)) {
        return { ok: true, files: [] }
      }
      const entries = fs.readdirSync(presetsDir, { withFileTypes: true })
      const files = entries
        .filter((e) => e.isFile())
        .map((e) => e.name)
      return { ok: true, files }
    } catch (e: any) {
      return { ok: false, error: e.message, files: [] }
    }
  })

  ipcMain.handle('rag:ingest', async (_event, filename: string) => {
    try {
      const filePath = path.join(repoRoot, 'sky_v1', 'rag', 'presets', filename)
      const resp = await fetch(`http://127.0.0.1:${backendPort}/v1/rag/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_path: filePath })
      })
      const text = await resp.text()
      try {
        return { ok: resp.ok, data: JSON.parse(text) }
      } catch {
        return { ok: resp.ok, data: text }
      }
    } catch (e: any) {
      return { ok: false, error: e.message }
    }
  })

  ipcMain.handle('terminal:spawn', (_event, shell?: string, cols: number = 80, rows: number = 24) => {
    if (ptyProcess) {
      ptyProcess.kill()
      ptyProcess = null
    }
    const defaultShell = process.platform === 'win32' ? 'powershell.exe' : process.env.SHELL || 'bash'
    const useShell = shell || defaultShell
    try {
      ptyProcess = ptySpawn(useShell, [], {
        name: 'xterm-256color',
        cols,
        rows,
        cwd: repoRoot,
        env: process.env as any
      })
      ptyProcess.onData((data) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('terminal:data', data)
        }
      })
      ptyProcess.onExit((ev) => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('terminal:exit', ev)
        }
        ptyProcess = null
      })
      return { ok: true, pid: ptyProcess.pid }
    } catch (e: any) {
      return { ok: false, error: e.message }
    }
  })

  ipcMain.handle('terminal:input', (_event, data: string) => {
    if (ptyProcess) {
      ptyProcess.write(data)
      return true
    }
    return false
  })

  ipcMain.handle('terminal:kill', () => {
    if (ptyProcess) {
      ptyProcess.kill()
      ptyProcess = null
      return true
    }
    return false
  })

  ipcMain.handle('terminal:resize', (_event, cols: number, rows: number) => {
    if (ptyProcess) {
      try {
        ptyProcess.resize(cols, rows)
        return true
      } catch {
        return false
      }
    }
    return false
  })
}

function setupAutoUpdater() {
  autoUpdater.autoDownload = false
  autoUpdater.autoInstallOnAppQuit = true

  autoUpdater.on('update-available', (info) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-available', info)
    }
  })

  autoUpdater.on('update-not-available', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-not-available')
    }
  })

  autoUpdater.on('download-progress', (progress) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:download-progress', progress)
    }
  })

  autoUpdater.on('update-downloaded', () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-downloaded')
    }
  })

  autoUpdater.on('error', (err) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('app:update-error', err.message)
    }
  })

  if (!app.isPackaged) return

  try {
    autoUpdater.checkForUpdates().catch(() => {})
  } catch {}
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: 'Sky V1 Omni Desktop',
    icon: path.join(__dirname, '..', '..', 'build', 'icon.png'),
    webPreferences: {
      preload: path.resolve(__dirname, '..', 'preload', 'index.js'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      webSecurity: true
    },
    backgroundColor: '#1e1e1e',
    show: false
  })

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
  })

  if (!app.isPackaged && process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(path.resolve(__dirname, '..', 'renderer', 'index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

app.whenReady().then(() => {
  setupIpc()
  createWindow()
  setupAutoUpdater()
  startBackend()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  stopBackend()
  if (ptyProcess) {
    ptyProcess.kill()
    ptyProcess = null
  }
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('before-quit', () => {
  stopBackend()
  if (ptyProcess) {
    ptyProcess.kill()
    ptyProcess = null
  }
})

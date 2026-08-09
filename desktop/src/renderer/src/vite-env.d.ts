/// <reference types="vite/client" />
import type { SkyWindow } from '../../preload/index'

declare global {
  interface Window {
    sky: SkyWindow
  }
}

export {}

import React, { useState, useEffect, useRef } from 'react'
import { Box, Loader2, Download, Server, ChevronDown, CheckCircle, RotateCcw, Eye, EyeOff, Code, Sparkles } from 'lucide-react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ProviderConfig, ProvidersStore } from '../../preload/index'

interface SceneObject {
  type: string
  position?: [number, number, number]
  rotation?: [number, number, number]
  scale?: [number, number, number] | number
  color?: string
  metalness?: number
  roughness?: number
  wireframe?: boolean
}

interface SceneJson {
  objects?: SceneObject[]
  background?: string
  ground?: boolean
  groundColor?: string
}

function createGeometry(type: string): THREE.BufferGeometry {
  const t = (type || 'box').toLowerCase()
  switch (t) {
    case 'sphere': return new THREE.SphereGeometry(0.5, 32, 32)
    case 'cylinder': return new THREE.CylinderGeometry(0.4, 0.4, 1, 32)
    case 'cone': return new THREE.ConeGeometry(0.5, 1, 32)
    case 'torus': return new THREE.TorusGeometry(0.4, 0.15, 16, 64)
    case 'plane': return new THREE.PlaneGeometry(1, 1)
    case 'dodecahedron': return new THREE.DodecahedronGeometry(0.5)
    case 'icosahedron': return new THREE.IcosahedronGeometry(0.5)
    case 'octahedron': return new THREE.OctahedronGeometry(0.5)
    case 'tetrahedron': return new THREE.TetrahedronGeometry(0.6)
    case 'torusknot': return new THREE.TorusKnotGeometry(0.35, 0.12, 100, 16)
    default: return new THREE.BoxGeometry(1, 1, 1)
  }
}

export default function ThreeDView() {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const objectsRef = useRef<THREE.Mesh[]>([])
  const animFrameRef = useRef<number>(0)

  const [prompt, setPrompt] = useState('')
  const [loading, setLoading] = useState(false)
  const [streamText, setStreamText] = useState('')
  const [showStream, setShowStream] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [store, setStore] = useState<ProvidersStore | null>(null)
  const [wireframeAll, setWireframeAll] = useState(false)
  const [sceneInfo, setSceneInfo] = useState<string>('')

  useEffect(() => {
    window.sky.providers.getStore().then((s) => setStore(s)).catch(() => {})
    const off = window.sky.media.on3DDelta(({ full }) => {
      setStreamText(full)
    })
    return () => { off() }
  }, [])

  // 初始化 Three.js
  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#1a1a2e')
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 100)
    camera.position.set(5, 5, 8)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // 光照
    const ambient = new THREE.AmbientLight(0xffffff, 0.4)
    scene.add(ambient)
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(5, 10, 7)
    dirLight.castShadow = true
    dirLight.shadow.mapSize.width = 2048
    dirLight.shadow.mapSize.height = 2048
    dirLight.shadow.camera.near = 0.5
    dirLight.shadow.camera.far = 50
    dirLight.shadow.camera.left = -10
    dirLight.shadow.camera.right = 10
    dirLight.shadow.camera.top = 10
    dirLight.shadow.camera.bottom = -10
    scene.add(dirLight)
    const hemiLight = new THREE.HemisphereLight(0x88aaff, 0x443322, 0.3)
    scene.add(hemiLight)

    // 网格地面
    const grid = new THREE.GridHelper(20, 20, 0x444466, 0x222233)
    ;(grid.material as THREE.Material).transparent = true
    ;(grid.material as THREE.Material).opacity = 0.5
    scene.add(grid)

    // 控制器
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 2
    controls.maxDistance = 30
    controls.maxPolarAngle = Math.PI / 2 + 0.2
    controlsRef.current = controls

    // 动画循环
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    // 默认场景：放一个示例立方体
    loadScene({
      objects: [
        { type: 'box', position: [0, 0.5, 0], color: '#4ec9b0', metalness: 0.3, roughness: 0.4 },
        { type: 'sphere', position: [-2, 0.5, 1], color: '#569cd6', metalness: 0.5, roughness: 0.3 },
        { type: 'cone', position: [2, 0.5, -1], color: '#dcdcaa', metalness: 0.2, roughness: 0.6 }
      ],
      background: '#1a1a2e',
      ground: true,
      groundColor: '#2a2a3e'
    })

    // 响应式
    const onResize = () => {
      if (!containerRef.current) return
      const w = containerRef.current.clientWidth
      const h = containerRef.current.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    const ro = new ResizeObserver(onResize)
    ro.observe(container)

    return () => {
      cancelAnimationFrame(animFrameRef.current)
      ro.disconnect()
      controls.dispose()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  const clearScene = () => {
    const scene = sceneRef.current
    if (!scene) return
    for (const obj of objectsRef.current) {
      scene.remove(obj)
      obj.geometry.dispose()
      if (Array.isArray(obj.material)) {
        obj.material.forEach((m) => m.dispose())
      } else {
        obj.material.dispose()
      }
    }
    objectsRef.current = []
  }

  const loadScene = (sceneJson: SceneJson) => {
    const scene = sceneRef.current
    if (!scene) return
    clearScene()

    // 背景
    if (sceneJson.background) {
      try { scene.background = new THREE.Color(sceneJson.background) } catch {}
    }

    // 地面
    if (sceneJson.ground) {
      const groundGeo = new THREE.PlaneGeometry(20, 20)
      const groundMat = new THREE.MeshStandardMaterial({
        color: sceneJson.groundColor || '#2a2a3e',
        roughness: 0.8,
        metalness: 0.1
      })
      const ground = new THREE.Mesh(groundGeo, groundMat)
      ground.rotation.x = -Math.PI / 2
      ground.position.y = 0
      ground.receiveShadow = true
      scene.add(ground)
      objectsRef.current.push(ground)
    }

    // 物体
    const objs = sceneJson.objects || []
    for (const o of objs) {
      try {
        const geo = createGeometry(o.type)
        const mat = new THREE.MeshStandardMaterial({
          color: o.color || '#cccccc',
          metalness: o.metalness ?? 0.3,
          roughness: o.roughness ?? 0.5,
          wireframe: o.wireframe || wireframeAll
        })
        const mesh = new THREE.Mesh(geo, mat)
        if (o.position) mesh.position.set(o.position[0], o.position[1], o.position[2])
        if (o.rotation) mesh.rotation.set(o.rotation[0], o.rotation[1], o.rotation[2])
        if (o.scale) {
          if (typeof o.scale === 'number') mesh.scale.setScalar(o.scale)
          else if (Array.isArray(o.scale)) mesh.scale.set(o.scale[0], o.scale[1], o.scale[2])
        }
        mesh.castShadow = true
        mesh.receiveShadow = true
        scene.add(mesh)
        objectsRef.current.push(mesh)
      } catch (e) {
        // 跳过无效物体
      }
    }
    setSceneInfo(`已加载 ${objs.length} 个物体`)
  }

  const toggleWireframe = () => {
    const next = !wireframeAll
    setWireframeAll(next)
    for (const obj of objectsRef.current) {
      const mat = obj.material as THREE.MeshStandardMaterial
      if (mat && 'wireframe' in mat) mat.wireframe = next
    }
  }

  const resetCamera = () => {
    const cam = cameraRef.current
    const ctrl = controlsRef.current
    if (cam && ctrl) {
      cam.position.set(5, 5, 8)
      ctrl.target.set(0, 0, 0)
      ctrl.update()
    }
  }

  const volcCfg: ProviderConfig | undefined = store?.providers?.volcengine
  const volcReady = !!(volcCfg?.apiKey && volcCfg?.model)

  const generate3D = async () => {
    if (!prompt.trim() || loading) return
    if (!volcReady) {
      setError('请先在「设置 → 模型服务商 → 火山引擎方舟」配置 API Key 和推理接入点 Endpoint ID')
      return
    }
    setLoading(true)
    setError(null)
    setStreamText('')
    setShowStream(true)
    try {
      const result = await window.sky.media.generate3D({
        apiKey: volcCfg!.apiKey,
        prompt: prompt.trim(),
        model: volcCfg!.model,
        baseUrl: volcCfg!.baseUrl
      })
      if (result.ok) {
        if (result.scene) {
          loadScene(result.scene as SceneJson)
          setSceneInfo(`字节推理流生成完成，已渲染 ${result.scene.objects?.length || 0} 个物体`)
        } else {
          setError('推理完成但未解析出有效 JSON 场景，请查看流式输出')
        }
      } else {
        setError(result.error || '生成失败')
      }
    } catch (e: any) {
      setError(e.message || String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div style={{ padding: 12, borderBottom: '1px solid var(--border-color)', background: 'var(--bg-sidebar)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
          <Box size={20} color="var(--fg-accent)" />
          <span style={{ fontWeight: 600 }}>3D Model Viewer</span>
          <span style={{ fontSize: 11, color: 'var(--fg-secondary)', marginLeft: 8 }}>
            字节火山在线推理流 → 场景 JSON → Three.js 渲染
          </span>
          <div style={{ flex: 1 }} />
          {volcReady ? (
            <span style={{ color: 'var(--success)', display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
              <CheckCircle size={12} /> 火山已配置
            </span>
          ) : (
            <span style={{ color: 'var(--warning)', fontSize: 12 }}>需先配置火山 API Key + Endpoint ID</span>
          )}
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end' }}>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate3D() } }}
            placeholder="描述你想生成的 3D 场景，例如：一个红色立方体放在蓝色球体旁边，地面是绿色的 (Ctrl+Enter 生成)"
            rows={2}
            style={{ flex: 1, minHeight: 48, padding: '8px 12px' }}
          />
          <button
            onClick={generate3D}
            disabled={loading || !prompt.trim()}
            style={{ height: 48, padding: '0 16px', background: 'var(--accent)', border: 'none', color: '#fff', display: 'flex', alignItems: 'center', gap: 6 }}
          >
            {loading ? <Loader2 size={16} style={{ animation: 'spin 0.8s linear infinite' }} /> : <Sparkles size={16} />}
            {loading ? '推理中…' : 'AI 生成场景'}
          </button>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* 3D 画布 */}
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
          {/* 工具栏 */}
          <div style={{
            position: 'absolute', top: 10, right: 10,
            display: 'flex', gap: 6, background: 'rgba(30,30,30,0.85)',
            padding: 6, borderRadius: 6, border: '1px solid var(--border-color)'
          }}>
            <button onClick={resetCamera} title="重置相机" style={{ padding: '4px 8px', background: 'transparent', border: 'none', color: 'var(--fg-primary)' }}>
              <RotateCcw size={15} />
            </button>
            <button onClick={toggleWireframe} title="线框模式" style={{ padding: '4px 8px', background: 'transparent', border: 'none', color: wireframeAll ? 'var(--accent)' : 'var(--fg-primary)' }}>
              {wireframeAll ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
            <button onClick={() => setShowStream(!showStream)} title="显示/隐藏推理流" style={{ padding: '4px 8px', background: 'transparent', border: 'none', color: showStream ? 'var(--accent)' : 'var(--fg-primary)' }}>
              <Code size={15} />
            </button>
          </div>
          {/* 状态信息 */}
          {sceneInfo && (
            <div style={{
              position: 'absolute', bottom: 10, left: 10,
              padding: '4px 10px', background: 'rgba(30,30,30,0.85)',
              borderRadius: 4, border: '1px solid var(--border-color)',
              fontSize: 11, color: 'var(--success)'
            }}>
              {sceneInfo}
            </div>
          )}
          {/* 操作提示 */}
          <div style={{
            position: 'absolute', bottom: 10, right: 10,
            padding: '4px 10px', background: 'rgba(30,30,30,0.85)',
            borderRadius: 4, border: '1px solid var(--border-color)',
            fontSize: 11, color: 'var(--fg-secondary)'
          }}>
            鼠标拖拽旋转 · 滚轮缩放 · 右键平移
          </div>
        </div>

        {/* 推理流输出面板 */}
        {showStream && (
          <div style={{ width: 380, borderLeft: '1px solid var(--border-color)', background: 'var(--bg-editor)', display: 'flex', flexDirection: 'column' }}>
            <div style={{
              padding: '8px 12px', borderBottom: '1px solid var(--border-color)',
              background: 'var(--bg-sidebar)', fontSize: 11, textTransform: 'uppercase',
              color: 'var(--fg-secondary)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6
            }}>
              <Code size={13} /> 字节推理流输出
              {loading && <Loader2 size={12} style={{ animation: 'spin 0.8s linear infinite' }} />}
            </div>
            <div style={{ flex: 1, overflow: 'auto', padding: 12 }}>
              <pre style={{
                fontFamily: 'Consolas, Monaco, monospace', fontSize: 11,
                color: '#9cdcfe', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                lineHeight: 1.5
              }}>
                {streamText || (loading ? '等待推理流开始…' : '点击「AI 生成场景」后，字节推理流的 JSON 输出将实时显示在此。')}
              </pre>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div style={{ padding: 10, borderTop: '1px solid var(--error)', background: 'rgba(244,135,113,0.1)', color: 'var(--error)', fontSize: 12 }}>
          {error}
        </div>
      )}
    </div>
  )
}

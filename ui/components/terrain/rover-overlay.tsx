"use client"

interface RoverOverlayProps {
  position: [number, number] | null
  heading: number
  width: number
  height: number
}

export function RoverOverlay({ position, heading, width, height }: RoverOverlayProps) {
  if (!position) return null

  const [row, col] = position
  const size = 10
  const rot = (heading * Math.PI) / 180

  return (
    <div
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ aspectRatio: `${width} / ${height}` }}
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-full block"
      >
        <g
          transform={`translate(${col}, ${row}) rotate(${rot})`}
        >
          {/* Body - rounded rect (top-down rover) */}
          <rect
            x={-size}
            y={-size * 0.7}
            width={size * 2}
            height={size * 1.4}
            rx={3}
            fill="#1e293b"
            stroke="#334155"
            strokeWidth={1}
          />
          {/* Cabin / front sensor */}
          <rect
            x={-size * 0.4}
            y={-size * 1.2}
            width={size * 0.8}
            height={size * 0.5}
            rx={1}
            fill="#334155"
          />
          {/* Wheels - 4 corners */}
          {[
            [-size * 0.85, -size * 0.85],
            [size * 0.85, -size * 0.85],
            [-size * 0.85, size * 0.85],
            [size * 0.85, size * 0.85],
          ].map(([wx, wy], i) => (
            <circle key={i} cx={wx} cy={wy} r={2.5} fill="#64748b" stroke="#475569" strokeWidth={0.5} />
          ))}
          {/* Direction indicator - front */}
          <polygon
            points={`0,${-size * 1.4} -3,${-size * 0.9} 3,${-size * 0.9}`}
            fill="#f59e0b"
            stroke="#d97706"
            strokeWidth={0.5}
          />
        </g>
      </svg>
    </div>
  )
}

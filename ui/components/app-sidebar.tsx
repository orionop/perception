"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import {
  ScanSearch,
  Swords,
  Navigation,
  Settings,
  ChevronLeft,
  ChevronRight,
  Satellite,
  FlaskConical,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { useState } from "react"

const navItems = [
  { label: "Command Center", href: "/technical", icon: FlaskConical },
  { label: "Perception Lab", href: "/perception", icon: ScanSearch },
  { label: "Model Arena", href: "/arena", icon: Swords },
]

export function AppSidebar() {
  const pathname = usePathname()
  const [collapsed, setCollapsed] = useState(false)

  return (
    <aside
      className={cn(
        "relative z-10 flex flex-col h-screen border-r border-border bg-sidebar shrink-0",
        collapsed ? "w-14" : "w-56"
      )}
      style={{ transition: "width 200ms ease-in-out" }}
    >
      <div
        className={cn(
          "flex items-center gap-3 px-3 h-14 border-b border-border shrink-0",
          collapsed && "justify-center px-0"
        )}
      >
        <div className="flex items-center justify-center w-8 h-8 shrink-0">
          <Satellite className="w-5 h-5 text-primary" />
        </div>
        {!collapsed && (
          <div className="flex flex-col min-w-0">
            <span className="text-xs font-bold text-sidebar-foreground tracking-wide uppercase truncate">
              Perception Engine
            </span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <div className="status-dot status-dot-green" />
              <span className="text-[9px] text-muted-foreground uppercase tracking-wider truncate">
                System Active
              </span>
            </div>
          </div>
        )}
      </div>

      <nav className="flex-1 py-3 px-1.5 space-y-0.5 overflow-y-auto">
        {navItems.map((item) => {
          const isActive =
            pathname === item.href ||
            (item.href !== "/" && pathname.startsWith(item.href))
          return (
            <Link
              key={item.href}
              href={item.href}
              prefetch={true}
              className={cn(
                "relative flex items-center gap-2.5 px-2.5 py-2 rounded text-xs font-medium transition-colors duration-150",
                collapsed && "justify-center px-0",
                isActive
                  ? "bg-primary/10 text-primary border border-primary/20"
                  : "text-muted-foreground hover:text-sidebar-foreground hover:bg-sidebar-accent border border-transparent"
              )}
            >
              <item.icon className={cn("w-4 h-4 shrink-0", isActive && "text-primary")} />
              {!collapsed && <span className="truncate tracking-wide">{item.label}</span>}
            </Link>
          )
        })}
      </nav>

      <div className="p-1.5 border-t border-border shrink-0">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="flex items-center justify-center w-full gap-2 px-2.5 py-1.5 text-xs text-muted-foreground hover:text-sidebar-foreground rounded hover:bg-sidebar-accent transition-colors duration-150"
        >
          {collapsed ? (
            <ChevronRight className="w-3.5 h-3.5" />
          ) : (
            <>
              <ChevronLeft className="w-3.5 h-3.5" />
              <span className="tracking-wide">Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  )
}

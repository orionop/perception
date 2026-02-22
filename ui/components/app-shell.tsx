"use client"

import { AppSidebar } from "@/components/app-sidebar"

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen">
      <AppSidebar />
      <main className="flex-1 overflow-y-auto min-w-0">
        <div className="p-6 lg:p-8">{children}</div>
      </main>
    </div>
  )
}

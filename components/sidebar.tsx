'use client'

import { Home, BookOpen, Settings } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface SidebarProps {
  currentPage: 'home' | 'dashboard'
  setCurrentPage: (page: 'home' | 'dashboard') => void
}

export default function Sidebar({ currentPage, setCurrentPage }: SidebarProps) {
  const navItems = [
    { id: 'home', label: 'Home', icon: Home },
    { id: 'dashboard', label: 'Study Dashboard', icon: BookOpen },
  ]

  return (
    <aside className="w-64 bg-gradient-to-b from-sidebar to-sidebar/80 text-sidebar-foreground border-r border-sidebar-border/30 min-h-screen flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-sidebar-border/30">
        <div className="text-2xl font-bold bg-gradient-to-r from-sidebar-primary to-sidebar-accent bg-clip-text text-transparent">
          Study Hub
        </div>
        <p className="text-xs text-sidebar-foreground/70 mt-1">NCERT Learning Platform</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon
          return (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id as any)}
              className={cn(
                'w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors',
                currentPage === item.id
                  ? 'bg-sidebar-primary text-sidebar-primary-foreground'
                  : 'text-sidebar-foreground hover:bg-sidebar-accent/50'
              )}
            >
              <Icon className="h-5 w-5" />
              {item.label}
            </button>
          )
        })}
      </nav>

      {/* Settings */}
      <div className="p-4 border-t border-sidebar-border/30">
        <Button
          variant="outline"
          className="w-full justify-start text-sidebar-foreground border-sidebar-border/30 hover:bg-sidebar-accent/50"
        >
          <Settings className="h-4 w-4 mr-2" />
          Settings
        </Button>
      </div>
    </aside>
  )
}

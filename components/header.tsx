import { Sun, Moon } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function Header() {
  const toggleTheme = () => {
    if (typeof window !== 'undefined') {
      document.documentElement.classList.toggle('dark')
    }
  }

  return (
    <header className="sticky top-0 bg-gradient-to-r from-primary to-accent border-b border-border/30 px-6 py-4 backdrop-blur-sm">
      <div className="flex justify-between items-center max-w-7xl mx-auto">
        <div>
          <h1 className="text-2xl font-bold text-white bg-gradient-to-r from-white to-white/80 bg-clip-text">
            NCERT Smart Study Buddy
          </h1>
        </div>
        <Button
          variant="outline"
          size="icon"
          onClick={toggleTheme}
          className="rounded-full bg-white/10 hover:bg-white/20 border-white/20 text-white"
        >
          <Sun className="h-5 w-5 dark:hidden" />
          <Moon className="h-5 w-5 hidden dark:block" />
        </Button>
      </div>
    </header>
  )
}

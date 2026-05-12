'use client'

import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Calculator, Microscope, Globe, BookOpen } from 'lucide-react'

interface DashboardProps {
  onStartLearning: () => void
}

export default function Dashboard({ onStartLearning }: DashboardProps) {
  const subjects = [
    { name: 'Mathematics', icon: Calculator, color: 'from-blue-400 to-blue-600' },
    { name: 'Science', icon: Microscope, color: 'from-blue-500 to-blue-700' },
    { name: 'Social Science', icon: Globe, color: 'from-blue-600 to-blue-800' },
    { name: 'English', icon: BookOpen, color: 'from-blue-400 to-blue-700' },
  ]

  return (
    <div className="min-h-full bg-gradient-to-b from-background via-background to-background dark:from-gray-950 dark:via-gray-900 dark:to-gray-950">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-20 text-center">
        <div className="mb-12">
          <div className="inline-block mb-6 px-4 py-2 bg-gradient-to-r from-primary/20 to-accent/20 border border-primary/30 rounded-full">
            <p className="text-sm font-semibold text-primary">Futuristic Learning Platform</p>
          </div>
          <h1 className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent mb-6">
            Welcome to Smart Study
          </h1>
          <p className="text-xl text-foreground/70 max-w-2xl mx-auto leading-relaxed">
            Generate flashcards, Q&A, and quizzes from NCERT textbooks for any class (1-10)
          </p>
        </div>

        <Button
          onClick={onStartLearning}
          size="lg"
          className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white px-8 py-6 text-lg rounded-full font-semibold shadow-lg"
        >
          Start Learning Now
        </Button>
      </div>

      {/* Subject Cards */}
      <div className="container mx-auto px-4 py-16">
        <h2 className="text-4xl font-bold text-foreground mb-12 text-center">Choose Your Subject</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
          {subjects.map((subject, index) => {
            const Icon = subject.icon
            return (
              <Card
                key={index}
                className="p-6 cursor-pointer hover:shadow-2xl transition-all hover:scale-105 border border-border/50 hover:border-primary/50 bg-gradient-to-br from-card to-card/80 group"
                onClick={onStartLearning}
              >
                <div className={`w-12 h-12 bg-gradient-to-br ${subject.color} rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-bold text-lg text-foreground group-hover:text-primary transition-colors">{subject.name}</h3>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Features Section */}
      <div className="container mx-auto px-4 py-20">
        <h2 className="text-4xl font-bold text-foreground mb-12 text-center">Features</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <Card className="p-8 border border-border/50 hover:border-primary/50 bg-gradient-to-br from-card to-card/80 hover:shadow-2xl transition-all group">
            <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-primary mb-3">Flashcards</h3>
            <p className="text-foreground/70 leading-relaxed">
              Automatically generate interactive flashcards from any chapter to master concepts
            </p>
          </Card>

          <Card className="p-8 border border-border/50 hover:border-primary/50 bg-gradient-to-br from-card to-card/80 hover:shadow-2xl transition-all group">
            <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-primary mb-3">Q&A</h3>
            <p className="text-foreground/70 leading-relaxed">
              Get important questions and detailed answers with explanations
            </p>
          </Card>

          <Card className="p-8 border border-border/50 hover:border-primary/50 bg-gradient-to-br from-card to-card/80 hover:shadow-2xl transition-all group">
            <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <BookOpen className="w-6 h-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-primary mb-3">Quizzes</h3>
            <p className="text-foreground/70 leading-relaxed">
              Test your knowledge with adaptive quizzes and track your progress
            </p>
          </Card>
        </div>
      </div>

      {/* CTA Section */}
      <div className="container mx-auto px-4 py-20 text-center">
        <Card className="p-12 bg-gradient-to-r from-primary/90 to-accent/90 text-white border-0 hover:shadow-2xl transition-all">
          <h2 className="text-4xl font-bold mb-4">Ready to Excel</h2>
          <p className="text-lg mb-8 text-white/90">
            Choose your class and subject to get started
          </p>
          <Button
            onClick={onStartLearning}
            size="lg"
            className="bg-white hover:bg-white/90 text-primary px-8 py-6 text-lg rounded-full font-bold shadow-lg"
          >
            Begin Studying
          </Button>
        </Card>
      </div>
    </div>
  )
}

'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Loader2, Download, RotateCw, Star } from 'lucide-react'
import { generateFlashcards, type FlashcardItem } from '@/lib/backend'

interface FlashcardsTabProps {
  classNum: number
  subject: string
  chapter: string | null
}

export default function FlashcardsTab({ classNum, subject, chapter }: FlashcardsTabProps) {
  const [numCards, setNumCards] = useState(5)
  const [difficulty, setDifficulty] = useState('medium')
  const [isLoading, setIsLoading] = useState(false)
  const [flipped, setFlipped] = useState<Set<number>>(new Set())
  const [starred, setStarred] = useState<Set<number>>(new Set())
  const [generated, setGenerated] = useState(false)
  const [cards, setCards] = useState<FlashcardItem[]>([])
  const [notes, setNotes] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleGenerate = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await generateFlashcards({
        class_num: classNum,
        subject: subject || null,
        chapter: chapter || null,
        count: numCards,
      })
      if (response.flashcards.length === 0) {
        setError('No flashcards were generated for this selection. Try another chapter or ingest the textbook content again.')
        setGenerated(false)
        return
      }
      setCards(response.flashcards)
      setNotes(response.notes)
      setGenerated(true)
      setFlipped(new Set())
      setStarred(new Set())
    } catch (generateError) {
      setError(generateError instanceof Error ? generateError.message : 'Failed to generate flashcards')
    } finally {
      setIsLoading(false)
    }
  }

  const toggleFlip = (id: number) => {
    const newFlipped = new Set(flipped)
    if (newFlipped.has(id)) {
      newFlipped.delete(id)
    } else {
      newFlipped.add(id)
    }
    setFlipped(newFlipped)
  }

  const toggleStar = (id: number) => {
    const newStarred = new Set(starred)
    if (newStarred.has(id)) {
      newStarred.delete(id)
    } else {
      newStarred.add(id)
    }
    setStarred(newStarred)
  }

  if (!generated) {
    return (
      <Card className="p-8 border border-border/50 bg-gradient-to-br from-card to-card/80">
        <div className="space-y-6">
          <div>
            <Label className="text-base font-semibold text-foreground">Number of Flashcards: {numCards}</Label>
            <Slider
              value={[numCards]}
              onValueChange={(val) => setNumCards(val[0])}
              min={5}
              max={20}
              step={1}
              className="mt-4"
            />
          </div>

          <div>
            <Label className="text-base font-semibold mb-4 block text-foreground">Difficulty Level</Label>
            <RadioGroup value={difficulty} onValueChange={setDifficulty}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="easy" id="easy" />
                <Label htmlFor="easy" className="text-foreground">Easy</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="medium" id="medium" />
                <Label htmlFor="medium" className="text-foreground">Medium</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="hard" id="hard" />
                <Label htmlFor="hard" className="text-foreground">Hard</Label>
              </div>
            </RadioGroup>
          </div>

          <Button
            onClick={handleGenerate}
            disabled={isLoading}
            className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white py-6 text-lg font-semibold"
          >
            {isLoading && <Loader2 className="mr-2 h-5 w-5 animate-spin" />}
            {isLoading ? 'Generating Flashcards' : 'Generate Flashcards'}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
          Your Flashcards ({cards.length})
        </h3>
        <div className="flex gap-3">
          <Button variant="outline" size="sm" onClick={() => setGenerated(false)} className="border-primary/50 hover:bg-primary/10">
            <RotateCw className="h-4 w-4 mr-2" />
            Regenerate
          </Button>
          <Button variant="outline" size="sm" className="border-primary/50 hover:bg-primary/10">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {notes.length > 0 && (
        <Card className="p-4 border border-primary/20 bg-primary/5 text-sm text-foreground/80">
          {notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {cards.map((card, index) => (
          <div key={`${card.front}-${index}`}>
            <Card
              className={`min-h-48 cursor-pointer transition-all transform p-6 border-2 flex flex-col justify-between ${
                flipped.has(index)
                  ? 'border-accent/60 bg-gradient-to-br from-accent/10 to-accent/5'
                  : 'border-primary/40 bg-gradient-to-br from-primary/10 to-primary/5 hover:shadow-lg hover:border-primary/60'
              }`}
              onClick={() => toggleFlip(index)}
            >
              <div>
                <p className="text-xs text-foreground/60 mb-3 font-semibold uppercase tracking-wide">
                  {flipped.has(index) ? 'Answer' : 'Question'}
                </p>
                <p className="text-lg font-semibold text-foreground leading-relaxed">
                  {flipped.has(index) ? card.back : card.front}
                </p>
                {flipped.has(index) && <p className="mt-3 text-sm text-foreground/70">{card.explanation}</p>}
              </div>
              <button
                className="self-end mt-4"
                onClick={(e) => {
                  e.stopPropagation()
                  toggleStar(index)
                }}
              >
                <Star
                  className={`h-6 w-6 transition-colors ${
                    starred.has(index)
                      ? 'fill-primary text-primary'
                      : 'text-foreground/20 hover:text-primary/60'
                  }`}
                />
              </button>
            </Card>
            <p className="text-xs text-foreground/50 mt-2 text-center">Click to flip</p>
          </div>
        ))}
      </div>
    </div>
  )
}

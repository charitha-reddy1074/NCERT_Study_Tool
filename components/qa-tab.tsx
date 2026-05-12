'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Loader2, Copy } from 'lucide-react'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { generateQuestions, type QuestionItem } from '@/lib/backend'

interface QATabProps {
  classNum: number
  subject: string
  chapter: string | null
}

export default function QATab({ classNum, subject, chapter }: QATabProps) {
  const [numQA, setNumQA] = useState('10')
  const [isLoading, setIsLoading] = useState(false)
  const [generated, setGenerated] = useState(false)
  const [items, setItems] = useState<QuestionItem[]>([])
  const [notes, setNotes] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleGenerate = async () => {
    if (!chapter) {
      setError('Please select a chapter first to generate chapter-specific Q&A.')
      return
    }

    setIsLoading(true)
    setError(null)
    try {
      const response = await generateQuestions({
        class_num: classNum,
        subject: subject || null,
        chapter: chapter || null,
        count: Number(numQA),
      })
      setItems(response.questions)
      setNotes(response.notes)
      setGenerated(true)
    } catch (generateError) {
      setError(generateError instanceof Error ? generateError.message : 'Failed to generate Q&A')
    } finally {
      setIsLoading(false)
    }
  }

  if (!generated) {
    return (
      <Card className="p-8 border border-border/50 bg-gradient-to-br from-card to-card/80">
        <div className="space-y-6">
          <div>
            <Label className="text-base font-semibold mb-4 block text-foreground">Number of Q&A Pairs</Label>
            <RadioGroup value={numQA} onValueChange={setNumQA}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="5" id="qa-5" />
                <Label htmlFor="qa-5" className="text-foreground">5 Q&A</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="10" id="qa-10" />
                <Label htmlFor="qa-10" className="text-foreground">10 Q&A</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="15" id="qa-15" />
                <Label htmlFor="qa-15" className="text-foreground">15 Q&A</Label>
              </div>
            </RadioGroup>
          </div>

          <Button
            onClick={handleGenerate}
            disabled={isLoading || !chapter}
            className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white py-6 text-lg font-semibold"
          >
            {isLoading && <Loader2 className="mr-2 h-5 w-5 animate-spin" />}
            {isLoading ? 'Generating Q&A' : 'Generate Important Q&A'}
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
          Questions & Answers ({items.length})
        </h3>
        <Button variant="outline" size="sm" className="border-primary/50 hover:bg-primary/10">
          <Copy className="h-4 w-4 mr-2" />
          Copy All
        </Button>
      </div>

      {notes.length > 0 && (
        <Card className="p-4 border border-primary/20 bg-primary/5 text-sm text-foreground/80">
          {notes.map((note) => (
            <p key={note}>{note}</p>
          ))}
        </Card>
      )}

      <Accordion type="single" collapsible className="space-y-2">
        {items.map((item, index) => (
          <AccordionItem key={`${item.question}-${index}`} value={`item-${index}`} className="border border-primary/30 rounded-lg px-4 bg-gradient-to-r from-card to-card/70">
            <AccordionTrigger className="hover:no-underline font-semibold text-foreground hover:text-primary transition-colors">
              <span className="text-left">{item.question}</span>
            </AccordionTrigger>
            <AccordionContent className="text-foreground/80 pt-4">
              <div className="bg-gradient-to-r from-primary/5 to-accent/5 p-4 rounded-lg border border-primary/20">
                <p className="mb-4 leading-relaxed">{item.answer}</p>
                <p className="mb-4 text-sm text-foreground/60">{item.explanation}</p>
                <Button size="sm" variant="outline" className="border-primary/50 hover:bg-primary/10">
                  <Copy className="h-3 w-3 mr-2" />
                  Copy
                </Button>
              </div>
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  )
}

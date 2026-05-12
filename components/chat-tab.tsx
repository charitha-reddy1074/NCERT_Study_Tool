'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Send, Loader2, Sparkles } from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { askQuestion, generateChapterSummary, type ChatMessage } from '@/lib/backend'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface ChatTabProps {
  classNum: number
  subject: string
  chapter: string | null
}

const INITIAL_MESSAGES: Message[] = [
  {
    id: '1',
    role: 'assistant',
    content: 'Hi! I am your NCERT Study Assistant. Ask me any question about the selected textbook chapter and I will answer from the official content.',
    timestamp: new Date(),
  },
]

export default function ChatTab({ classNum, subject, chapter }: ChatTabProps) {
  const [messages, setMessages] = useState<Message[]>(INITIAL_MESSAGES)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSend = async () => {
    if (!input.trim() || isLoading) return

    // Add user message
    const userMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setError(null)

    const chatHistory: ChatMessage[] = messages.map((message) => ({
      role: message.role,
      content: message.content,
    }))

    try {
      const response = await askQuestion({
        class_num: classNum,
        subject: subject || null,
        chapter: chapter || null,
        question: input,
        chat_history: chatHistory,
      })

      const assistantMessage: Message = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (sendError) {
      setError(sendError instanceof Error ? sendError.message : 'Failed to get an answer from the backend')
    } finally {
      setIsLoading(false)
    }
  }

  const handleChapterSummary = async () => {
    if (isLoading || isSummarizing || !chapter) return

    setIsSummarizing(true)
    setError(null)

    try {
      const response = await generateChapterSummary({
        class_num: classNum,
        subject: subject || null,
        chapter: chapter || null,
      })

      const summaryMessage: Message = {
        id: `summary-${Date.now()}`,
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, summaryMessage])
    } catch (summaryError) {
      setError(summaryError instanceof Error ? summaryError.message : 'Failed to generate chapter summary')
    } finally {
      setIsSummarizing(false)
    }
  }

  return (
    <Card className="flex flex-col h-96 md:h-screen max-h-96 md:max-h-none border border-border/50 bg-gradient-to-b from-card to-card/80">
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs md:max-w-md lg:max-w-lg px-4 py-3 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-gradient-to-r from-primary to-accent text-white rounded-br-none shadow-md'
                    : 'bg-gradient-to-r from-accent/10 to-primary/10 text-foreground rounded-bl-none border border-primary/20'
                }`}
              >
                <p className="text-sm leading-relaxed">{message.content}</p>
                <p className="text-xs mt-2 opacity-60">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gradient-to-r from-accent/10 to-primary/10 text-foreground px-4 py-3 rounded-lg rounded-bl-none border border-primary/20">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="border-t border-border/30 p-4 bg-gradient-to-r from-background to-primary/5">
        {error && <p className="mb-3 text-xs text-destructive">{error}</p>}
        <div className="flex gap-2">
          <Input
            placeholder="Ask your question here..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            disabled={isLoading}
            className="flex-1 border-primary/30 focus:border-primary/60"
          />
          <Button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            size="icon"
            className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white"
          >
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
        <div className="mt-3 flex gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={handleChapterSummary}
            disabled={!chapter || isLoading || isSummarizing}
            className="border-primary/40 hover:bg-primary/10"
          >
            {isSummarizing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
            Summarize Chapter
          </Button>
        </div>
        <p className="text-xs text-foreground/50 mt-2">
          Powered by the backend RAG pipeline • {subject || 'Any subject'}{chapter ? ` • ${chapter}` : ''}
        </p>
      </div>
    </Card>
  )
}

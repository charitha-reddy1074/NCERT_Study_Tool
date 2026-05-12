'use client'

import { useEffect, useMemo, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import Header from '@/components/header'
import Sidebar from '@/components/sidebar'
import FlashcardsTab from '@/components/flashcards-tab'
import QATab from '@/components/qa-tab'
import QuizTab from '@/components/quiz-tab'
import ChatTab from '@/components/chat-tab'
import Dashboard from '@/components/dashboard'
import { getCatalog, ingestSelection, type CatalogResponse, type CatalogSubject } from '@/lib/backend'

type Page = 'home' | 'dashboard'

export default function Page() {
  const [currentPage, setCurrentPage] = useState<Page>('home')
  const [classNum, setClassNum] = useState('6')
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null)
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(false)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [selectedSubject, setSelectedSubject] = useState('')
  const [selectedChapter, setSelectedChapter] = useState('')
  const [isIngesting, setIsIngesting] = useState(false)
  const [ingestError, setIngestError] = useState<string | null>(null)

  useEffect(() => {
    let isMounted = true
    const loadCatalog = async () => {
      setIsLoadingCatalog(true)
      setCatalogError(null)
      try {
        const response = await getCatalog(Number(classNum))
        if (!isMounted) return
        setCatalog(response)
        setSelectedSubject((current) => {
          const availableSubjects = response.subjects
          if (availableSubjects.length === 0) return ''
          const currentExists = availableSubjects.some((subject) => subject.key === current)
          return currentExists ? current : availableSubjects[0].key
        })
      } catch (error) {
        if (!isMounted) return
        setCatalog(null)
        setCatalogError(error instanceof Error ? error.message : 'Failed to load catalog')
      } finally {
        if (isMounted) setIsLoadingCatalog(false)
      }
    }

    void loadCatalog()
    return () => {
      isMounted = false
    }
  }, [classNum])

  useEffect(() => {
    setSelectedChapter('all')
  }, [selectedSubject])

  const selectedSubjectData = useMemo<CatalogSubject | null>(() => {
    if (!catalog) return null
    return catalog.subjects.find((subject) => subject.key === selectedSubject) ?? null
  }, [catalog, selectedSubject])

  const chapter = selectedChapter && selectedChapter !== 'all' ? selectedChapter : null

  // On-demand ingestion: when a concrete chapter is selected, ingest that
  // single chapter immediately (no separate ingest button, no auto-ingest on
  // class/subject change). This is a fully lazy, chapter-level ingest.
  useEffect(() => {
    let isMounted = true
    if (currentPage !== 'dashboard' || !chapter || !selectedSubject) return

    const ingestChapter = async () => {
      setIsIngesting(true)
      setIngestError(null)
      try {
        await ingestSelection({
          class_num: Number(classNum),
          subject: selectedSubject,
          chapter: chapter,
          clear_existing: true,
        })
      } catch (error) {
        if (!isMounted) return
        setIngestError(error instanceof Error ? error.message : 'Failed to ingest selected NCERT chapter')
      } finally {
        if (isMounted) setIsIngesting(false)
      }
    }

    void ingestChapter()
    return () => {
      isMounted = false
    }
  }, [chapter, classNum, selectedSubject, currentPage])

  return (
    <div className="flex min-h-screen bg-background">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <div className="flex-1 flex flex-col">
        <Header />
        <main className="flex-1 overflow-auto">
          {currentPage === 'home' && <Dashboard onStartLearning={() => setCurrentPage('dashboard')} />}
          {currentPage === 'dashboard' && (
            <div className="container mx-auto px-4 py-8">
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-semibold text-foreground mb-2 block">Class</label>
                    <Select value={classNum} onValueChange={setClassNum} disabled={isIngesting}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select Class" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from({ length: 10 }, (_, i) => i + 1).map((classNum) => (
                          <SelectItem key={classNum} value={String(classNum)}>
                            Class {classNum}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-foreground mb-2 block">Subject</label>
                    <Select value={selectedSubject} onValueChange={setSelectedSubject} disabled={isLoadingCatalog || isIngesting}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select Subject" />
                      </SelectTrigger>
                      <SelectContent>
                        {(catalog?.subjects ?? []).map((subject) => (
                          <SelectItem key={subject.key} value={subject.key}>
                            {subject.label} ({subject.chapter_count})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-foreground mb-2 block">Chapter</label>
                    <Select value={selectedChapter} onValueChange={setSelectedChapter} disabled={isLoadingCatalog || isIngesting || !selectedSubjectData}>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select Chapter" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All chapters</SelectItem>
                        {(selectedSubjectData?.chapters ?? []).map((chapter) => (
                          <SelectItem key={chapter.key} value={chapter.key}>
                            {chapter.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Ingestion is triggered automatically when a concrete chapter is selected. */}

                {isLoadingCatalog && (
                  <div className="flex items-center gap-2 text-sm text-foreground/70">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading NCERT catalog from the backend...
                  </div>
                )}

                {catalogError && (
                  <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
                    Backend catalog error: {catalogError}
                  </div>
                )}

                {isIngesting && (
                  <div className="flex items-center gap-2 text-sm text-foreground/70">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Ingesting NCERT files for class {classNum}, {selectedSubject}...
                  </div>
                )}

                {ingestError && (
                  <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive">
                    Backend ingest error: {ingestError}
                  </div>
                )}

                {!chapter && (
                  <div className="rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-sm text-foreground/80">
                    1. Select a subject, then select a chapter — the chapter PDF will be ingested on selection. 2. Wait a few seconds for ingestion to finish. 3. Choose a tab to start learning with flashcards, Q&A, quizzes, or ask AI.
                  </div>
                )}

                <Tabs defaultValue="flashcards" className="w-full">
                  <TabsList className="grid w-full grid-cols-4">
                    <TabsTrigger value="flashcards" disabled={isIngesting || !chapter}>Flashcards</TabsTrigger>
                    <TabsTrigger value="qa" disabled={isIngesting || !chapter}>Q&A</TabsTrigger>
                    <TabsTrigger value="quiz" disabled={isIngesting || !chapter}>Quiz</TabsTrigger>
                    <TabsTrigger value="chat" disabled={isIngesting || !chapter}>Ask AI</TabsTrigger>
                  </TabsList>

                  <TabsContent value="flashcards" className="mt-6">
                    <FlashcardsTab
                      classNum={Number(classNum)}
                      subject={selectedSubject}
                      chapter={chapter}
                    />
                  </TabsContent>

                  <TabsContent value="qa" className="mt-6">
                    <QATab classNum={Number(classNum)} subject={selectedSubject} chapter={chapter} />
                  </TabsContent>

                  <TabsContent value="quiz" className="mt-6">
                    <QuizTab classNum={Number(classNum)} subject={selectedSubject} chapter={chapter} />
                  </TabsContent>

                  <TabsContent value="chat" className="mt-6">
                    <ChatTab classNum={Number(classNum)} subject={selectedSubject} chapter={chapter} />
                  </TabsContent>
                </Tabs>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export type ChatMessage = {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export type CatalogChapter = {
  key: string
  label: string
  file_path?: string | null
}

export type CatalogSubject = {
  key: string
  label: string
  chapter_count: number
  chapters: CatalogChapter[]
}

export type CatalogResponse = {
  class_num: number
  subjects: CatalogSubject[]
  source_dir: string
  note?: string | null
}

export type SourceCitation = {
  source_path: string
  file_name: string
  class_num?: number | null
  subject?: string | null
  chapter?: string | null
  page?: number | null
  chunk_id?: number | null
  relevance_score?: number | null
  excerpt?: string | null
}

export type ChatResponse = {
  answer: string
  citations: SourceCitation[]
  not_in_textbook: boolean
  retrieved_documents: number
  source_scope?: string | null
}

export type ChapterSummaryResponse = ChatResponse

export type FlashcardItem = {
  front: string
  back: string
  explanation: string
}

export type FlashcardResponse = {
  flashcards: FlashcardItem[]
  citations: SourceCitation[]
  notes: string[]
  not_in_textbook: boolean
}

export type QuestionItem = {
  question: string
  answer: string
  explanation: string
  difficulty: 'easy' | 'medium' | 'hard'
}

export type QuestionResponse = {
  questions: QuestionItem[]
  citations: SourceCitation[]
  notes: string[]
  not_in_textbook: boolean
}

export type QuizOption = {
  label: string
  text: string
}

export type QuizItem = {
  question: string
  question_type: 'mcq' | 'true_false' | 'short_answer'
  options: QuizOption[]
  correct_answer: string
  explanation: string
  difficulty: 'easy' | 'medium' | 'hard'
}

export type QuizResponse = {
  quiz_title: string
  questions: QuizItem[]
  citations: SourceCitation[]
  notes: string[]
  not_in_textbook: boolean
}

export type IngestResponse = {
  source_dir: string
  files_processed: number
  chunks_indexed: number
  collection_size: number
  skipped_files: string[]
  note?: string | null
}

const BACKEND_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BACKEND_BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Request failed with status ${response.status}`)
  }

  return response.json() as Promise<T>
}

export async function getCatalog(classNum: number): Promise<CatalogResponse> {
  return requestJson<CatalogResponse>(`/api/v1/catalog?class_num=${classNum}`)
}

export async function askQuestion(payload: {
  class_num: number
  subject?: string | null
  chapter?: string | null
  question: string
  chat_history?: ChatMessage[]
  top_k?: number
}): Promise<ChatResponse> {
  return requestJson<ChatResponse>('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({
      top_k: 6,
      chat_history: [],
      ...payload,
    }),
  })
}

export async function generateChapterSummary(payload: {
  class_num: number
  subject?: string | null
  chapter?: string | null
  top_k?: number
}): Promise<ChapterSummaryResponse> {
  return requestJson<ChapterSummaryResponse>('/api/v1/summary', {
    method: 'POST',
    body: JSON.stringify({
      top_k: 6,
      ...payload,
    }),
  })
}

export async function generateFlashcards(payload: {
  class_num: number
  subject?: string | null
  chapter?: string | null
  focus_area?: string | null
  count?: number
  top_k?: number
}): Promise<FlashcardResponse> {
  return requestJson<FlashcardResponse>('/api/v1/flashcards', {
    method: 'POST',
    body: JSON.stringify({
      count: 10,
      top_k: 6,
      ...payload,
    }),
  })
}

export async function generateQuestions(payload: {
  class_num: number
  subject?: string | null
  chapter?: string | null
  count?: number
  top_k?: number
}): Promise<QuestionResponse> {
  return requestJson<QuestionResponse>('/api/v1/questions', {
    method: 'POST',
    body: JSON.stringify({
      count: 10,
      top_k: 6,
      ...payload,
    }),
  })
}

export async function generateQuiz(payload: {
  class_num: number
  subject?: string | null
  chapter?: string | null
  quiz_type?: 'mcq' | 'mixed' | 'true_false' | 'short_answer'
  difficulty?: 'easy' | 'medium' | 'hard'
  count?: number
  top_k?: number
}): Promise<QuizResponse> {
  return requestJson<QuizResponse>('/api/v1/quiz', {
    method: 'POST',
    body: JSON.stringify({
      quiz_type: 'mcq',
      difficulty: 'medium',
      count: 10,
      top_k: 6,
      ...payload,
    }),
  })
}

export async function ingestSelection(payload: {
  class_num: number
  subject?: string | null
  chapter?: string | null
  source_dir?: string | null
  clear_existing?: boolean
}): Promise<IngestResponse> {
  return requestJson<IngestResponse>('/api/v1/ingest', {
    method: 'POST',
    body: JSON.stringify({
      clear_existing: true,
      ...payload,
    }),
  })
}

'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Loader2, BarChart3 } from 'lucide-react'
import { generateQuiz, type QuizItem } from '@/lib/backend'

interface QuizTabProps {
  classNum: number
  subject: string
  chapter: string | null
}

export default function QuizTab({ classNum, subject, chapter }: QuizTabProps) {
  const [numQuestions, setNumQuestions] = useState('10')
  const [difficulty, setDifficulty] = useState('medium')
  const [quizType, setQuizType] = useState<'mcq' | 'mixed' | 'true_false'>('mcq')
  const [isLoading, setIsLoading] = useState(false)
  const [quizStarted, setQuizStarted] = useState(false)
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selected, setSelected] = useState('')
  const [shortAnswer, setShortAnswer] = useState('')
  const [answers, setAnswers] = useState<string[]>([])
  const [completed, setCompleted] = useState(false)
  const [questions, setQuestions] = useState<QuizItem[]>([])
  const [quizTitle, setQuizTitle] = useState('NCERT Quiz')
  const [error, setError] = useState<string | null>(null)

  const handleStart = async () => {
    if (!chapter) {
      setError('Please select a chapter first to generate chapter-specific quiz questions.')
      return
    }

    setIsLoading(true)
    setError(null)
    try {
      const response = await generateQuiz({
        class_num: classNum,
        subject: subject || null,
        chapter: chapter || null,
        quiz_type: quizType,
        difficulty: difficulty as 'easy' | 'medium' | 'hard',
        count: Number(numQuestions),
      })
      if (response.questions.length === 0) {
        setError('No quiz questions were generated for this chapter. Try again or choose another chapter.')
        setQuizStarted(false)
        return
      }
      setQuestions(response.questions)
      setQuizTitle(response.quiz_title)
      setAnswers(Array(response.questions.length).fill(''))
      setQuizStarted(true)
      setCurrentQuestion(0)
      setSelected('')
      setShortAnswer('')
      setCompleted(false)
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : 'Failed to start quiz')
    } finally {
      setIsLoading(false)
    }
  }

  const handleNext = () => {
    const newAnswers = [...answers]
    newAnswers[currentQuestion] = questions[currentQuestion]?.question_type === 'short_answer' ? shortAnswer : selected
    setAnswers(newAnswers)

    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1)
      const nextAnswer = answers[currentQuestion + 1] ?? ''
      setSelected(nextAnswer)
      setShortAnswer(nextAnswer)
    } else {
      setCompleted(true)
    }
  }

  const handlePrevious = () => {
    const newAnswers = [...answers]
    newAnswers[currentQuestion] = questions[currentQuestion]?.question_type === 'short_answer' ? shortAnswer : selected
    setAnswers(newAnswers)
    setCurrentQuestion(currentQuestion - 1)
    const previousAnswer = answers[currentQuestion - 1] ?? ''
    setSelected(previousAnswer)
    setShortAnswer(previousAnswer)
  }

  const correctCount = answers.reduce((count, answer, idx) => {
    const question = questions[idx]
    if (!question) return count
    const normalizedAnswer = answer.trim().toLowerCase()
    const normalizedCorrect = question.correct_answer.trim().toLowerCase()
    const optionMatch = question.options.find((option) => option.label.toLowerCase() === normalizedCorrect || option.text.trim().toLowerCase() === normalizedCorrect)
    const selectedMatchesCorrect =
      normalizedAnswer === normalizedCorrect ||
      question.options.some((option) => option.label.toLowerCase() === normalizedAnswer && option.text.trim().toLowerCase() === normalizedCorrect)
    return count + (selectedMatchesCorrect || (optionMatch ? normalizedAnswer === optionMatch.label.toLowerCase() || normalizedAnswer === optionMatch.text.trim().toLowerCase() : false) ? 1 : 0)
  }, 0)

  const score = questions.length > 0 ? Math.round((correctCount / questions.length) * 100) : 0

  if (!quizStarted) {
    return (
      <Card className="p-8 border border-border/50 bg-gradient-to-br from-card to-card/80">
        <div className="space-y-6">
          <div>
            <Label className="text-base font-semibold mb-4 block text-foreground">Quiz Type</Label>
            <RadioGroup value={quizType} onValueChange={(value) => setQuizType(value as 'mcq' | 'mixed' | 'true_false')}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="mcq" id="mcq" />
                <Label htmlFor="mcq" className="text-foreground">Multiple Choice</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="mixed" id="mixed" />
                <Label htmlFor="mixed" className="text-foreground">Mixed Questions</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="true_false" id="true_false" />
                <Label htmlFor="true_false" className="text-foreground">True/False</Label>
              </div>
            </RadioGroup>
          </div>

          <div>
            <Label className="text-base font-semibold mb-4 block text-foreground">Number of Questions</Label>
            <RadioGroup value={numQuestions} onValueChange={setNumQuestions}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="10" id="q-10" />
                <Label htmlFor="q-10" className="text-foreground">10 Questions</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="15" id="q-15" />
                <Label htmlFor="q-15" className="text-foreground">15 Questions</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="20" id="q-20" />
                <Label htmlFor="q-20" className="text-foreground">20 Questions</Label>
              </div>
            </RadioGroup>
          </div>

          <div>
            <Label className="text-base font-semibold mb-4 block text-foreground">Difficulty Level</Label>
            <RadioGroup value={difficulty} onValueChange={setDifficulty}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="easy" id="d-easy" />
                <Label htmlFor="d-easy" className="text-foreground">Easy</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="medium" id="d-medium" />
                <Label htmlFor="d-medium" className="text-foreground">Medium</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="hard" id="d-hard" />
                <Label htmlFor="d-hard" className="text-foreground">Hard</Label>
              </div>
            </RadioGroup>
          </div>

          <Button
            onClick={handleStart}
            disabled={isLoading || !chapter}
            className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white py-6 text-lg font-semibold"
          >
            {isLoading && <Loader2 className="mr-2 h-5 w-5 animate-spin" />}
            {isLoading ? 'Starting Quiz' : 'Start Quiz'}
          </Button>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
      </Card>
    )
  }

  if (completed) {
    return (
      <div className="space-y-6">
        <Card className="p-12 bg-gradient-to-br from-primary/20 to-accent/20 border-2 border-primary/40 text-center">
          <h2 className="text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-6">Quiz Completed</h2>
          <div className="mb-8">
            <p className="text-7xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent mb-4">{score}%</p>
            <p className="text-xl text-foreground/70">
              You got {correctCount} out of {questions.length} correct
            </p>
          </div>
          <div className="flex gap-4 justify-center">
            <Button
              onClick={() => {
                setQuizStarted(false)
                setCompleted(false)
              }}
              className="bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white"
            >
              Retake Quiz
            </Button>
            <Button variant="outline" className="border-primary/50 hover:bg-primary/10">Review Answers</Button>
          </div>
        </Card>

        <Card className="p-6 border border-border/50 bg-gradient-to-br from-card to-card/80">
          <h3 className="text-xl font-bold text-foreground mb-4 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            Answer Summary
          </h3>
          <div className="space-y-3">
            {questions.map((q, idx) => (
              <div key={`${q.question}-${idx}`} className="flex items-center justify-between p-3 bg-gradient-to-r from-primary/5 to-accent/5 rounded-lg border border-primary/20">
                <span className="text-foreground font-medium">Question {idx + 1}</span>
                <span
                  className={`text-sm font-bold ${
                    answers[idx]?.trim().toLowerCase() === q.correct_answer.trim().toLowerCase() ? 'text-primary' : 'text-accent'
                  }`}
                >
                  {answers[idx]?.trim().toLowerCase() === q.correct_answer.trim().toLowerCase() ? 'Correct' : 'Incorrect'}
                </span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    )
  }

  const question = questions[currentQuestion]

  if (!question) {
    return (
      <Card className="p-8 border border-border/50 bg-gradient-to-br from-card to-card/80">
        <p className="text-sm text-foreground/70">No quiz questions were returned for this selection.</p>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      <Card className="p-6 border border-primary/40 bg-gradient-to-br from-card to-card/80">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-3">
            <p className="text-sm font-semibold text-foreground/70">
              {quizTitle} • Question {currentQuestion + 1} of {questions.length}
            </p>
            <p className="text-sm text-foreground/70">60 seconds</p>
          </div>
          <div className="h-2 bg-gradient-to-r from-primary to-accent rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-accent transition-all"
              style={{ width: `${questions.length > 0 ? ((currentQuestion + 1) / questions.length) * 100 : 0}%` }}
            />
          </div>
        </div>

        <h3 className="text-2xl font-bold text-foreground mb-8 leading-relaxed">{question.question}</h3>

        {question.question_type === 'short_answer' ? (
          <textarea
            value={shortAnswer}
            onChange={(event) => setShortAnswer(event.target.value)}
            placeholder="Type your answer here..."
            className="min-h-32 w-full rounded-lg border border-primary/30 bg-background px-4 py-3 text-foreground"
          />
        ) : (
          <RadioGroup value={selected} onValueChange={setSelected}>
            <div className="space-y-3">
              {question.options.map((option) => (
                <div key={option.label} className="flex items-center space-x-3 p-4 border-2 border-primary/30 rounded-lg hover:border-primary/60 cursor-pointer bg-gradient-to-r hover:from-primary/5 hover:to-accent/5 transition-all">
                  <RadioGroupItem value={option.label} id={`option-${option.label}`} />
                  <Label htmlFor={`option-${option.label}`} className="cursor-pointer flex-1 font-medium text-foreground">
                    {option.label}. {option.text}
                  </Label>
                </div>
              ))}
            </div>
          </RadioGroup>
        )}

        <p className="mt-4 text-sm text-foreground/60">{question.explanation}</p>
      </Card>

      <div className="flex justify-between gap-4">
        <Button
          onClick={handlePrevious}
          disabled={currentQuestion === 0}
          variant="outline"
          className="flex-1 border-primary/50 hover:bg-primary/10"
        >
          Previous
        </Button>
        <Button
          onClick={handleNext}
          disabled={question.question_type === 'short_answer' ? !shortAnswer.trim() : !selected}
          className="flex-1 bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-white font-semibold"
        >
          {currentQuestion === questions.length - 1 ? 'Complete' : 'Next'}
        </Button>
      </div>
    </div>
  )
}

**✅ Complete UI Prompt for Frontend Development**

You can copy-paste the prompt below directly to Cursor, Claude, GPT-4o, Grok, or any AI coding assistant to generate your frontend.

---

### **UI Development Prompt**

**Project Name:** NCERT Smart Study Assistant

**Tech Stack:** Python + **Streamlit** (preferred)

**Goal:**  
Build a clean, modern, colorful, and student-friendly web UI for a RAG-based learning assistant that generates flashcards, Q&A, and quizzes from NCERT textbooks across Classes 1-10.

---

### **Overall Design Requirements**

- Modern, clean, and playful design suitable for 11–12 year old students.
- Use bright but professional colors (blue, green, purple, orange accents).
- Responsive design (works well on laptop and tablet).
- Sidebar for navigation.
- Dark/Light mode toggle (default: Light mode).
- Use emojis liberally to make it engaging for kids.

---

### **Key Pages / Sections**

#### **1. Home / Landing Page**
- Big welcoming header: “NCERT Smart Study Buddy ✨”
- Subtitle: “Generate Flashcards, Quizzes & Answers from your textbooks”
- Quick subject cards (clickable) for:
  - Mathematics
  - Science
  - Social Science (History, Geography, Civics)
  - English
  - Hindi
- Recent Activity section (last generated items)
- “Start Learning” big button

#### **2. Main Study Dashboard (Core Page)**

**Left Sidebar:**
- Class: Dropdown selector (Classes 1-10)
- Subject Selector (Dropdown or clickable cards)
- Chapter Selector (Dynamic – loads based on selected subject)
- Topic Selector (Multi-select optional)

**Main Area (Tabbed Interface):**

**Tab 1: Generate Flashcards**
- Number of flashcards: Slider (5 to 20)
- Difficulty: Easy / Medium / Hard
- Focus Area: Whole Chapter / Specific Topics
- “Generate Flashcards” button
- Output: Beautiful flip-card style display
- Features: 
  - “Star” important cards
  - “Export to Anki” button
  - “Regenerate” button

**Tab 2: Q&A Generator**
- “Generate Important Questions & Answers”
- Number of Q&A: 5 / 10 / 15
- Output: Accordion style (Question → Click to reveal Answer + Explanation)

**Tab 3: Quiz Mode**
- Quiz Type: MCQ / Mixed / True-False / Short Answer
- Number of Questions: 10 / 15 / 20
- Difficulty Level
- “Start Quiz” button
- During quiz: Timer, Progress bar, Next/Previous
- At end: Score, Correct answers with explanations, Retry option

**Tab 4: Ask Anything (Chat)**
- Chat interface like ChatGPT
- User can ask any doubt related to the selected subject/chapter
- RAG-powered responses with source chapter reference

#### **3. Library / All Content Page**
- Search bar (across all books)
- Filter by Subject, Chapter, Topic
- Grid view of all generated sets (Flashcards, Quizzes)

---

### **Key Functionalities to Implement**

1. **Session State Management**
   - Store selected subject, chapter, generated content
   - Maintain chat history

2. **Backend Integration**
   - Connect with your LlamaIndex + Ollama backend via API or direct function calls
   - Show loading spinners with fun messages (“Reading your textbook…”, “Creating smart questions…”)

3. **Output Features**
   - Copy button for each card/question
   - Download as PDF / JSON
   - Export Flashcards as .csv or Anki (.apkg) format
   - Print option

4. **User Experience**
   - Progress indicator while generating
   - Success animations
   - Error handling with friendly messages
   - “I don’t know” or “Not in textbook” handling

5. **Extra Nice-to-Have Features**
   - Bookmark favorite flashcards
   - Spaced repetition suggestion (simple version)
   - Voice input for questions (optional)
   - Confetti on quiz completion (high score)

---

### **Design Specifications**

- Use `st.columns()` effectively
- Use `st.expander`, `st.tabs`, `st.popover`
- Card design using `st.container` + custom CSS
- Make flashcards visually appealing (front/back flip effect using custom CSS or two sides)
- Font: Clean sans-serif (use Google fonts if possible)
- Logo/Icon: Book + Sparkles emoji

---

**Deliverables Expected:**
- Complete Streamlit app in a single or multi-page format
- Well-organized code with comments
- Custom CSS for better UI polish
- Proper error handling and loading states

---

Would you like me to also give you:

1. **Ready-to-use Streamlit code skeleton** (right now)?
2. **Custom CSS** for this project?
3. **Backend API structure** that matches this UI?

Just tell me which one you want next.

# 🥗 AI Nutrition & Diet Planner

**Level:** Intermediate
**Core concept:** LangChain LLM Chain + custom calculator tool
**Stack:** Streamlit, LangChain, Groq / Google Gemini

## What it does
- Takes your age, gender, height, weight, activity level, goal, diet preference,
  allergies, and preferred cuisine.
- A custom Python tool (`tools.py`) deterministically calculates:
  - BMI (+ category)
  - BMR (Mifflin-St Jeor equation)
  - TDEE (activity-adjusted)
  - Daily calorie target (based on your goal)
  - Macro split (protein / carbs / fat in grams)
- Those numbers are passed into a LangChain `prompt | llm | parser` chain that
  generates a personalized daily meal plan, tips, and a disclaimer.
- Plan is downloadable as a `.txt` file.

## Why "LLM Chain + calculator tool"?
LLMs are unreliable at arithmetic. Instead of asking the model to compute BMI/
calories itself (which it will often get subtly wrong), we compute those with
real formulas in Python first, then hand the *numbers* to the LLM — so it only
does what it's good at: turning structured data into a friendly, readable plan.

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

You'll need an API key from either:
- [Groq](https://console.groq.com/keys) (free, fast — llama-3.3-70b-versatile)
- [Google AI Studio](https://aistudio.google.com/apikey) (Gemini)

Enter it in the sidebar — it's used only for your session, never stored.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI + LangChain chain |
| `tools.py` | Custom calculator tool (BMI/BMR/TDEE/macros) |
| `requirements.txt` | Dependencies |

## Possible extensions
- Turn the calculator into a proper LangChain `@tool` bound to the LLM via
  `bind_tools()` for full agentic tool-calling (ReAct-style), instead of the
  simpler "calculate first, then generate" chain used here.
- Add a 7-day plan with grocery list generation.
- Add Tavily search to pull in real recipe links (matches your other repos'
  pattern of using `tavily-python`).
- Persist plans per user with a simple SQLite/CSV log.

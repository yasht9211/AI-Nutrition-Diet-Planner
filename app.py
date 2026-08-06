"""
AI Nutrition & Diet Planner (v2)
---------------------------------
Level: Intermediate
Core concept: LangChain LLM Chain + a custom Python calculator tool
              + a conversational chat chain (chat with an AI nutritionist)
Stack: Streamlit + LangChain + Groq / Google Gemini

Flow:
1. User fills in profile (age, gender, height, weight, activity, goal, diet
   preference, allergies, cuisine).
2. The custom `nutrition_calculator` tool (tools.py) deterministically computes
   BMI / BMR / TDEE / calorie target / macros — no LLM guesswork on numbers.
3. Those numbers + preferences are fed into a LangChain prompt | llm chain,
   which generates a personalized daily meal plan and recommendations.
4. NEW: A chat panel lets the user talk to an "AI Nutritionist" — a
   conversational chain that remembers the chat history and, if a plan has
   already been generated, also knows the user's BMI/calorie/macro numbers
   so its answers stay consistent with the plan above.
"""

from __future__ import annotations

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import json
import re

from tools import calculate_nutrition, ACTIVITY_MULTIPLIERS, GOAL_ADJUSTMENT

st.set_page_config(page_title="AI Nutrition & Diet Planner", page_icon="🥗", layout="wide")

# ----------------------------- Background styling -----------------------------
# Uses your sunset river/landscape photo (assets/background.jpg, shipped
# alongside this app) as a fixed background so the text on top stays
# readable regardless of theme. Falls back to a plain look if the image
# file is missing, and tells you exactly why (wrong path in the repo is
# the #1 cause).
import base64
import os
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
# Try a couple of likely locations in case the repo structure differs slightly.
_CANDIDATE_PATHS = [
    _APP_DIR / "assets" / "background.jpg",
    _APP_DIR / "background.jpg",
    Path.cwd() / "assets" / "background.jpg",
]
BACKGROUND_IMAGE_PATH = next((p for p in _CANDIDATE_PATHS if p.exists()), None)


def _get_background_css() -> str:
    if BACKGROUND_IMAGE_PATH is not None:
        encoded = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode()
        return f"""
            background-image: linear-gradient(rgba(255,255,255,0), rgba(255,255,255,0)),
                               url('data:image/jpeg;base64,{encoded}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        """
    return """
        background-image: linear-gradient(180deg, #ffffff 0%, #fbfbf9 100%);
        background-attachment: fixed;
    """


_bg_css = _get_background_css()

st.markdown(
    f"""
    <style>
    .stApp,
    [data-testid="stAppViewContainer"] {{
        {_bg_css}
    }}
    [data-testid="stHeader"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="block-container"],
    .main .block-container {{
        background-color: transparent !important;
    }}

    /* Readable text over the photo, regardless of light/dark theme or
       which part of the image (bright sky vs dark mountains) sits behind
       it — dark, slightly bold text, no glow/highlight. */
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] div[data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] .stCaption {{
        color: #ffffff !important;
    }}
    [data-testid="stAppViewContainer"] h1 {{
        font-weight: 800 !important;
    }}
    /* Small text (captions, help text, field labels) bumped up for readability */
    [data-testid="stAppViewContainer"] .stCaption,
    [data-testid="stAppViewContainer"] small,
    [data-testid="stAppViewContainer"] label p {{
        font-size: 1rem !important;
    }}

    [data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {{
    background-color: rgba(255, 255, 255, 0.5) !important;
    background-image: none !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}}
    }}
    [data-testid="stSidebar"] * {{
        color: #262730 !important;
        text-shadow: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if BACKGROUND_IMAGE_PATH is None:
    st.warning(
        "⚠️ Background image not found. It should live at **assets/background.jpg** "
        "in your GitHub repo (same folder as app.py, inside an 'assets' subfolder). "
        f"Checked: {', '.join(str(p) for p in _CANDIDATE_PATHS)}"
    )

# ----------------------------- Sidebar: LLM setup -----------------------------
def get_secret(key: str) -> str:
    """Safely read a Streamlit secret; never raises if no secrets are configured."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


st.sidebar.title("⚙️ Model Settings")
provider = st.sidebar.selectbox("LLM Provider", ["Groq", "Google Gemini"])

# Try Streamlit Secrets first (App settings -> Secrets on Streamlit Cloud),
# fall back to manual entry if not configured.
if provider == "Groq":
    api_key = get_secret("GROQ_API_KEY")
    model_name = st.sidebar.selectbox(
        "Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )
else:
    api_key = get_secret("GOOGLE_API_KEY")
    model_name = st.sidebar.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-pro"])

if api_key:
    st.sidebar.success(f"✅ {provider} API key loaded from Secrets")
else:
    api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
    st.sidebar.caption(
        "Tip: add this once under App settings → Secrets on Streamlit Cloud "
        "so you never have to paste it again."
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Your API key is only used for this session and is never stored in the app."
)

# Tavily is used to fetch a representative photo for each meal in your plan.
tavily_key = get_secret("TAVILY_API_KEY") or st.sidebar.text_input(
    "Tavily API Key (for meal photos)", type="password",
    help="Free key from tavily.com — used to fetch a photo for each meal in your plan.",
)


def get_meal_image_url(meal_query: str) -> str | None:
    """Fetch one representative photo for a meal name via Tavily image search."""
    if not tavily_key:
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=tavily_key)
        res = client.search(query=f"{meal_query} food dish", include_images=True, max_results=1)
        images = res.get("images", [])
        return images[0] if images else None
    except Exception:
        return None


def extract_meal_names(plan_text: str, llm) -> list[str]:
    """Ask the LLM to pull out short, image-searchable dish names from the plan."""
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract only the distinct dish/meal names from the plan below. "
                "Return STRICT JSON: a list of short strings (2-5 words each), "
                "nothing else, no markdown, no explanation.",
            ),
            ("human", "{plan}"),
        ]
    )
    chain = extract_prompt | llm | StrOutputParser()
    raw = chain.invoke({"plan": plan_text})
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        names = json.loads(match.group(0))
        return [n for n in names if isinstance(n, str)][:8]  # cap to keep it fast
    except Exception:
        return []


def get_llm():
    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.4)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.4)


# session state — keeps the calculated plan & chat history across reruns
if "nutrition_result" not in st.session_state:
    st.session_state.nutrition_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of HumanMessage / AIMessage


# ----------------------------- Main UI -----------------------------
st.title("🥗 AI Nutrition & Diet Planner")
st.write(
    "Get your BMI, daily calorie target, and a personalized AI-generated "
    "meal plan — powered by a LangChain chain with a built-in calculator tool."
)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)
with col3:
    activity_level = st.selectbox("Activity Level", list(ACTIVITY_MULTIPLIERS.keys()), index=1)
    goal = st.selectbox("Goal", list(GOAL_ADJUSTMENT.keys()))

col4, col5 = st.columns(2)
with col4:
    diet_pref = st.selectbox(
        "Dietary Preference", ["No preference", "Vegetarian", "Vegan", "Non-vegetarian", "Eggetarian"]
    )
    cuisine = st.text_input("Preferred Cuisine (optional)", placeholder="e.g. Indian, Mediterranean")
with col5:
    allergies = st.text_input("Allergies / Foods to avoid (optional)", placeholder="e.g. peanuts, dairy")
    meals_per_day = st.slider("Meals per day", 3, 6, 4)

generate = st.button("🍽️ Generate My Plan", type="primary", use_container_width=True)

if generate:
    if not api_key:
        st.error(f"Please enter your {provider} API key in the sidebar.")
        st.stop()

    # Step 1: deterministic calculation via the custom tool
    with st.spinner("Calculating BMI, BMR, TDEE and macros..."):
        result = calculate_nutrition(
            age=age,
            gender=gender,
            height_cm=height_cm,
            weight_kg=weight_kg,
            activity_level=activity_level,
            goal=goal,
        )

    # Step 2: LangChain LLM Chain generates the personalized plan
    with st.spinner("Asking the AI nutritionist to build your meal plan..."):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a certified nutrition coach. Use ONLY the numeric "
                    "targets given to you (do not recalculate them). Be practical, "
                    "concise, and use simple, accessible foods. Always add a short "
                    "disclaimer that this is not medical advice.",
                ),
                (
                    "human",
                    """Build a {meals_per_day}-meal daily plan for this person:

- Age: {age}, Gender: {gender}
- BMI: {bmi} ({bmi_category})
- Daily calorie target: {calorie_target} kcal
- Macros: {protein_g} g protein / {carbs_g} g carbs / {fat_g} g fat
- Goal: {goal}
- Dietary preference: {diet_pref}
- Preferred cuisine: {cuisine}
- Allergies/avoid: {allergies}

Format the answer as:
1. A one-line summary of the strategy
2. A table-like breakdown of each meal with approximate calories and macros
3. 3 short personalized tips for hitting this goal
4. A one-line medical disclaimer""",
                ),
            ]
        )

        llm = get_llm()
        chain = prompt | llm | StrOutputParser()

        plan = chain.invoke(
            {
                "meals_per_day": meals_per_day,
                "age": age,
                "gender": gender,
                "bmi": result.bmi,
                "bmi_category": result.bmi_category,
                "calorie_target": result.calorie_target,
                "protein_g": result.protein_g,
                "carbs_g": result.carbs_g,
                "fat_g": result.fat_g,
                "goal": goal,
                "diet_pref": diet_pref,
                "cuisine": cuisine or "Any",
                "allergies": allergies or "None",
            }
        )

    # Step 3: extract meal names + fetch a photo for each (if Tavily key set)
    meal_images = {}
    if tavily_key:
        with st.spinner("Fetching meal photos..."):
            meal_names = extract_meal_names(plan, llm)
            for name in meal_names:
                url = get_meal_image_url(name)
                if url:
                    meal_images[name] = url

    # Save everything to session_state so it survives the rerun that
    # happens every time the user sends a chat message below.
    st.session_state.nutrition_result = {
        "result": result,
        "plan": plan,
        "meal_images": meal_images,
        "age": age,
        "gender": gender,
        "goal": goal,
        "diet_pref": diet_pref,
        "cuisine": cuisine or "Any",
        "allergies": allergies or "None",
    }
    st.session_state.chat_history = []  # reset chat when a new plan is made

# ----------------------------- Show plan (if generated) -----------------------------
if st.session_state.nutrition_result:
    data = st.session_state.nutrition_result
    result = data["result"]

    st.subheader("📊 Your Numbers")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", result.bmi, result.bmi_category)
    m2.metric("BMR", f"{result.bmr} kcal")
    m3.metric("TDEE", f"{result.tdee} kcal")
    m4.metric("Daily Target", f"{result.calorie_target} kcal")
    st.caption(
        f"Macro split for '{data['goal']}' → "
        f"Protein: {result.protein_g} g | Carbs: {result.carbs_g} g | Fat: {result.fat_g} g"
    )

    st.subheader("📋 Your Personalized Meal Plan")
    st.markdown(data["plan"])

    meal_images = data.get("meal_images", {})
    if meal_images:
        st.subheader("📸 Meal Photos")
        cols = st.columns(4)
        for i, (name, url) in enumerate(meal_images.items()):
            with cols[i % 4]:
                st.image(url, caption=name, use_container_width=True)
    elif not tavily_key:
        st.info("💡 Add a Tavily API key in the sidebar to see photos of each meal.")
    else:
        st.warning(
            "⚠️ Couldn't fetch meal photos this time — either the Tavily key is "
            "invalid/expired, or the free quota ran out. Check the key at "
            "tavily.com and try generating the plan again."
        )

    st.download_button(
        "⬇️ Download Plan as Text",
        data=data["plan"],
        file_name="my_nutrition_plan.txt",
        mime="text/plain",
    )

st.markdown("---")

# ============================================================
#            💬 CHAT WITH YOUR AI NUTRITIONIST
# ============================================================
st.subheader("💬 Chat with your AI Nutritionist")
st.caption(
    "Ask follow-up questions — swap a meal, adjust for a cheat day, ask "
    "about a specific food, etc. The nutritionist remembers this conversation "
    "and your numbers above (if you've generated a plan)."
)

# Render existing chat history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_msg = st.chat_input("Ask your AI nutritionist something...")

if user_msg:
    if not api_key:
        st.error(f"Please enter your {provider} API key in the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(user_msg)

    # Build context from the generated plan, if any, so the chatbot's
    # answers stay consistent with the numbers/plan shown above.
    if st.session_state.nutrition_result:
        d = st.session_state.nutrition_result
        r = d["result"]
        profile_context = (
            f"The user's profile: age {d['age']}, gender {d['gender']}, goal '{d['goal']}', "
            f"dietary preference {d['diet_pref']}, allergies/avoid {d['allergies']}. "
            f"Their calculated numbers: BMI {r.bmi} ({r.bmi_category}), daily calorie target "
            f"{r.calorie_target} kcal, macros {r.protein_g}g protein / {r.carbs_g}g carbs / "
            f"{r.fat_g}g fat. Their generated meal plan:\n{d['plan']}"
        )
    else:
        profile_context = (
            "The user has not generated a meal plan yet. Answer generally, and "
            "suggest they fill in the form above and click 'Generate My Plan' "
            "for personalized numbers."
        )

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a certified, friendly AI nutritionist having an ongoing "
                "chat with a client. Use the profile/context below ONLY as "
                "background — don't recalculate numbers yourself, refer back to "
                "the ones given. Keep answers conversational and concise. Always "
                "remind the user this is not a substitute for professional "
                "medical advice when relevant.\n\nContext:\n{profile_context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm = get_llm()
            chat_chain = chat_prompt | llm | StrOutputParser()
            response = chat_chain.invoke(
                {
                    "profile_context": profile_context,
                    "chat_history": st.session_state.chat_history,
                    "input": user_msg,
                }
            )
            st.markdown(response)

    st.session_state.chat_history.append(HumanMessage(content=user_msg))
    st.session_state.chat_history.append(AIMessage(content=response))

if st.session_state.chat_history:
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

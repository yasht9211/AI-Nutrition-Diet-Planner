"""
AI Nutrition & Diet Planner
----------------------------
Level: Intermediate
Core concept: LangChain LLM Chain + a custom Python calculator tool
Stack: Streamlit + LangChain + Groq / Google Gemini

Flow:
1. User fills in profile (age, gender, height, weight, activity, goal, diet
   preference, allergies, cuisine).
2. The custom `nutrition_calculator` tool (tools.py) deterministically computes
   BMI / BMR / TDEE / calorie target / macros — no LLM guesswork on numbers.
3. Those numbers + preferences are fed into a LangChain prompt | llm chain,
   which generates a personalized daily meal plan and recommendations.
"""

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import calculate_nutrition, ACTIVITY_MULTIPLIERS, GOAL_ADJUSTMENT

st.set_page_config(page_title="AI Nutrition & Diet Planner", page_icon="🥗", layout="wide")

# ----------------------------- Sidebar: LLM setup -----------------------------
st.sidebar.title("⚙️ Model Settings")
provider = st.sidebar.selectbox("LLM Provider", ["Groq", "Google Gemini"])

if provider == "Groq":
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    model_name = st.sidebar.selectbox(
        "Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    )
else:
    api_key = st.sidebar.text_input("Google API Key", type="password")
    model_name = st.sidebar.selectbox("Model", ["gemini-2.0-flash", "gemini-1.5-pro"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Your API key is only used for this session and is never stored."
)


def get_llm():
    if provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, api_key=api_key, temperature=0.4)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.4)


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

    # ----------------------------- Metrics -----------------------------
    st.subheader("📊 Your Numbers")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("BMI", result.bmi, result.bmi_category)
    m2.metric("BMR", f"{result.bmr} kcal")
    m3.metric("TDEE", f"{result.tdee} kcal")
    m4.metric("Daily Target", f"{result.calorie_target} kcal")

    st.caption(
        f"Macro split for '{goal}' → "
        f"Protein: {result.protein_g} g | Carbs: {result.carbs_g} g | Fat: {result.fat_g} g"
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

    st.subheader("📋 Your Personalized Meal Plan")
    st.markdown(plan)

    st.download_button(
        "⬇️ Download Plan as Text",
        data=plan,
        file_name="my_nutrition_plan.txt",
        mime="text/plain",
    )

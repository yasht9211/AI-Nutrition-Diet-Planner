"""
tools.py
--------
Custom Python tool used inside the LangChain chain: a deterministic
nutrition calculator (BMI, BMR, TDEE, calorie target, macro split).

Why a custom tool instead of letting the LLM "guess" the numbers?
LLMs are unreliable at arithmetic. We compute everything with real
formulas here, then hand the *numbers* to the LLM so it only has to
do what it's good at: turning structured data into a friendly,
personalized meal plan.
"""

from dataclasses import dataclass, asdict
from langchain_core.tools import Tool


ACTIVITY_MULTIPLIERS = {
    "Sedentary (little/no exercise)": 1.2,
    "Lightly active (1-3 days/week)": 1.375,
    "Moderately active (3-5 days/week)": 1.55,
    "Very active (6-7 days/week)": 1.725,
    "Extremely active (athlete/physical job)": 1.9,
}

GOAL_ADJUSTMENT = {
    "Lose weight": -500,   # ~0.5 kg/week deficit
    "Maintain weight": 0,
    "Gain weight": 400,    # lean surplus
}

GOAL_MACRO_SPLIT = {
    # (protein %, carb %, fat %)
    "Lose weight": (0.35, 0.35, 0.30),
    "Maintain weight": (0.25, 0.45, 0.30),
    "Gain weight": (0.30, 0.45, 0.25),
}


@dataclass
class NutritionResult:
    bmi: float
    bmi_category: str
    bmr: float
    tdee: float
    calorie_target: float
    protein_g: float
    carbs_g: float
    fat_g: float


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal weight"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def calculate_nutrition(
    age: int,
    gender: str,
    height_cm: float,
    weight_kg: float,
    activity_level: str,
    goal: str,
) -> NutritionResult:
    """Pure-Python calculator — no LLM involved. Deterministic and testable."""

    # BMI
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # BMR — Mifflin-St Jeor equation
    if gender == "Male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    # TDEE
    tdee = bmr * ACTIVITY_MULTIPLIERS[activity_level]

    # Calorie target based on goal
    calorie_target = tdee + GOAL_ADJUSTMENT[goal]
    calorie_target = max(calorie_target, 1200)  # safety floor

    # Macro split (grams): protein/carbs = 4 kcal/g, fat = 9 kcal/g
    protein_pct, carb_pct, fat_pct = GOAL_MACRO_SPLIT[goal]
    protein_g = (calorie_target * protein_pct) / 4
    carbs_g = (calorie_target * carb_pct) / 4
    fat_g = (calorie_target * fat_pct) / 9

    return NutritionResult(
        bmi=round(bmi, 1),
        bmi_category=_bmi_category(bmi),
        bmr=round(bmr),
        tdee=round(tdee),
        calorie_target=round(calorie_target),
        protein_g=round(protein_g),
        carbs_g=round(carbs_g),
        fat_g=round(fat_g),
    )


def _tool_func(query: str) -> str:
    """
    Adapter so the calculator can be invoked as a LangChain Tool with a
    single string input, e.g. "age=28,gender=Male,height_cm=175,weight_kg=78,
    activity_level=Moderately active (3-5 days/week),goal=Lose weight"
    """
    params = {}
    for part in query.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        params[k.strip()] = v.strip()

    result = calculate_nutrition(
        age=int(params["age"]),
        gender=params["gender"],
        height_cm=float(params["height_cm"]),
        weight_kg=float(params["weight_kg"]),
        activity_level=params["activity_level"],
        goal=params["goal"],
    )
    return str(asdict(result))


nutrition_calculator_tool = Tool(
    name="nutrition_calculator",
    func=_tool_func,
    description=(
        "Calculates BMI, BMR, TDEE, daily calorie target, and macro split "
        "(protein/carbs/fat in grams) from age, gender, height_cm, weight_kg, "
        "activity_level, and goal. Input must be a comma-separated string of "
        "key=value pairs."
    ),
)

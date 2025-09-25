import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import openai
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. OpenAI Setup (v0.28)
# -----------------------------
openai.api_key = "API"

# -----------------------------
# 2. Mock Data / Placeholder for Wearable API
# -----------------------------
def generate_mock_data(days_back=7):
    n_points = days_back * 24
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_points, freq="h")
    hr_df = pd.DataFrame({"timestamp": timestamps, "HR": np.random.normal(70, 10, n_points).astype(int)})
    hrv_df = pd.DataFrame({"timestamp": timestamps, "HRV": np.random.normal(60, 15, n_points)})
    steps_df = pd.DataFrame({"timestamp": timestamps, "steps": np.random.randint(0, 500, n_points)})
    sleep_df = pd.DataFrame({
        "date": [datetime.date.today() - datetime.timedelta(days=i) for i in range(days_back)][::-1],
        "light": np.random.randint(3,5,days_back),
        "deep": np.random.randint(2,4,days_back),
        "rem": np.random.randint(1,2,days_back),
        "duration_hours": np.random.randint(6,9,days_back)
    })
    cal_df = pd.DataFrame({"timestamp": timestamps, "calories": np.random.randint(50, 150, n_points)})
    return hr_df, hrv_df, steps_df, sleep_df, cal_df

hr_df, hrv_df, steps_df, sleep_df, cal_df = generate_mock_data()
weather = {"temp_C": 22, "humidity": 55, "conditions": "Sunny"}

# -----------------------------
# 3. User Profile Sidebar
# -----------------------------
st.sidebar.header("👤 User Profile")
age = st.sidebar.number_input("Age", 10, 100, 30)
fitness_level = st.sidebar.selectbox("Fitness Level", ["Beginner", "Intermediate", "Advanced"])
goal = st.sidebar.text_input("Goal", "Improve recovery")
conditions = st.sidebar.text_input("Chronic Conditions (optional)")
nutrition = st.sidebar.text_area("Today's Nutrition (manual input)")

user_profile = {"Age": age, "Fitness Level": fitness_level, "Goal": goal,
                "Chronic Conditions": conditions, "Nutrition": nutrition}

# -----------------------------
# 4. Compute Readiness (Current)
# -----------------------------
def compute_readiness(hr_df, hrv_df, sleep_df):
    avg_hr = hr_df["HR"].mean()
    avg_hrv = hrv_df["HRV"].mean()
    avg_sleep = sleep_df["duration_hours"].mean()
    readiness = (avg_hrv/80 + avg_sleep/8 - (avg_hr-70)/50) * 100
    return max(0, min(100, readiness))

readiness_score = compute_readiness(hr_df, hrv_df, sleep_df)

# -----------------------------
# 5. Multi-Day Forecast
# -----------------------------
def forecast_multi_day(hrv_df, sleep_df, steps_df, user_profile, days=5):
    X = np.arange(len(hrv_df)).reshape(-1,1)
    model_hrv = LinearRegression().fit(X, hrv_df["HRV"].values)
    forecast_scores = []
    for i in range(1, days+1):
        next_hrv = model_hrv.predict([[len(hrv_df)+i]])[0]
        next_sleep = sleep_df["duration_hours"].values[-1]
        forecast = (next_hrv/80 + next_sleep/8) * 100
        # Personalization
        if user_profile["Chronic Conditions"]:
            forecast *= 0.9
        if len(user_profile["Nutrition"]) < 20:
            forecast *= 0.95
        if user_profile["Fitness Level"] == "Beginner":
            forecast *= 0.95
        elif user_profile["Fitness Level"] == "Advanced":
            forecast *= 1.05
        forecast_scores.append(max(0, min(100, forecast)))
    return forecast_scores

forecast_5day = forecast_multi_day(hrv_df, sleep_df, steps_df, user_profile)

# -----------------------------
# 6. AI Interactive Dialogue with Biometrics Context
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def ai_response(message):
    # Include context of biometrics, readiness, and forecast
    context = f"""
User profile: {user_profile}
Current metrics:
- Average HR: {hr_df['HR'].mean():.1f} bpm
- Average HRV: {hrv_df['HRV'].mean():.1f} ms
- Steps today: {steps_df['steps'].sum():,}
- Sleep last night: {sleep_df['duration_hours'].values[-1]}h
- Calories burned today: {cal_df['calories'].sum():.0f} kcal
- Current readiness: {readiness_score:.0f}/100
Forecast for next 5 days: {['{:.0f}'.format(s) for s in forecast_5day]}
- Weather: {weather['temp_C']}°C, {weather['conditions']}, Humidity {weather['humidity']}%
"""
    history = "\n".join([f"User: {m['user']}\nAI: {m['ai']}" for m in st.session_state["chat_history"]])
    prompt = f"{context}\n{history}\nUser: {message}\nAI:"
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].text.strip()

user_input = st.text_input("Ask your AI coach a question:")
if user_input:
    answer = ai_response(user_input)
    st.session_state["chat_history"].append({"user": user_input, "ai": answer})
    st.write(f"💬 AI Coach: {answer}")

# -----------------------------
# 7. Streamlit Dashboard
# -----------------------------
st.title("🧬 Ultimate Bio-AI Dashboard (Contextual AI Feedback)")

tabs = st.tabs(["Trends", "Forecast", "Environment", "Notifications"])

with tabs[0]:
    st.subheader("📊 Multi-Day Trends")
    st.line_chart(hr_df.set_index("timestamp")["HR"])
    st.line_chart(hrv_df.set_index("timestamp")["HRV"])
    st.line_chart(steps_df.set_index("timestamp")["steps"])
    st.bar_chart(sleep_df.set_index("date")["duration_hours"])

with tabs[1]:
    st.subheader("📈 5-Day Personalized Forecast")
    days = [f"Day {i}" for i in range(1,6)]
    st.line_chart(pd.DataFrame({"Day": days, "Readiness": forecast_5day}).set_index("Day")["Readiness"])

with tabs[2]:
    st.subheader("🌤 Environment")
    st.write(f"Temperature: {weather['temp_C']}°C, Humidity: {weather['humidity']}%, Conditions: {weather['conditions']}")

with tabs[3]:
    st.subheader("🔔 Personalized Notifications")
    if readiness_score < 60:
        st.warning("⚠️ Low readiness today. Prioritize recovery.")
    for i, score in enumerate(forecast_5day, 1):
        if score < 60:
            st.warning(f"⚠️ Day {i} forecast: Low readiness forecasted. Adjust exercise and recovery.")

st.info("⚠️ Demo with mock data. AI answers include all biometric context. Not medical advice.")

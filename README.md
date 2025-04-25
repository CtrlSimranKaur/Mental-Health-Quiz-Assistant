# Mental-Health-Quiz-Assistant
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np



# ==========================
# Part 1: Data and CSV Export
# ==========================

data = [
    {"username": "Alex", "score": 18, "timestamp": "2025-04-23 14:15"},
    {"username": "Jordan", "score": 12, "timestamp": "2025-04-22 11:02"},
    {"username": "Sam", "score": 42, "timestamp": "2025-04-21 09:48"},
    {"username": "Jamie", "score": 35, "timestamp": "2025-04-20 17:30"},
    {"username": "Taylor", "score": 39, "timestamp": "2025-04-19 10:10"},
    {"username": "Morgan", "score": 65, "timestamp": "2025-04-18 13:45"},
    {"username": "Casey", "score": 46, "timestamp": "2025-04-17 15:20"},
    {"username": "Drew", "score": 30, "timestamp": "2025-04-16 19:40"},
    {"username": "Riley", "score": 35, "timestamp": "2025-04-15 08:55"},
    {"username": "Sky", "score": 44, "timestamp": "2025-04-14 14:00"},
     {"username": "Sky", "score": 10, "timestamp": "2025-04-14 14:00"},
    {"username": "Pat", "score": 36, "timestamp": "2025-04-13 09:25"},
    {"username": "Sam", "score": 49, "timestamp": "2025-04-12 16:30"},
    {"username": "Simran", "score": 34, "timestamp": "2025-04-16 19:40"},
    {"username": "Gary", "score": 67, "timestamp": "2025-04-15 08:55"},
    {"username": "Skylar", "score": 28, "timestamp": "2025-04-14 14:00"},
    {"username": "Patrick", "score": 16, "timestamp": "2025-04-13 09:25"},
    {"username": "ParPeterker", "score": 70, "timestamp": "2025-02-13 18:10"}
    
]




# Write to CSV
with open("quiz_data.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["username", "score", "timestamp"])
    writer.writeheader()
    writer.writerows(data)

print("✅ Data written to quiz_data.csv")

# ==========================
# Part 2: Load Data & Train ML Model
# ==========================

df = pd.DataFrame(data)

# Classify score into categories
def classify_score(score):
    if score >=41 :
        return 'High'
    elif score <= 40:
        return 'Low'
    else:
        return 'Medium'

df['class'] = df['score'].apply(classify_score)

# Split dataset
X = df[['score']]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Predict example
new_score = np.array([[56]])
predicted_class = model.predict(new_score)
print(f"🔮 Predicted class for score 56 : {predicted_class[0]}")

# ==========================
# Part 3: Mental Health Quiz
# ==========================

def mental_health_quiz():
    print("\n🧠 Welcome to the 20-Question Mental Health Quiz 🧠")
    print("Answer honestly. This quiz is for reflection, not diagnosis.\n")

    questions = [
        {"question": "1. How often do you feel overwhelmed?", "options": {"a": "Rarely", "b": "Sometimes", "c": "Often", "d": "Almost always"}},
        {"question": "2. How well are you sleeping lately?", "options": {"a": "Very well", "b": "Okay", "c": "Poorly", "d": "Barely sleeping"}},
        {"question": "3. How often do you feel anxious or nervous?", "options": {"a": "Rarely", "b": "Sometimes", "c": "Often", "d": "Almost always"}},
        {"question": "4. Do you find joy in activities you used to enjoy?", "options": {"a": "Yes, always", "b": "Most of the time", "c": "Sometimes", "d": "Rarely or never"}},
        {"question": "5. Do you feel connected to others?", "options": {"a": "Very connected", "b": "Somewhat connected", "c": "Isolated sometimes", "d": "Very isolated"}},
        {"question": "6. How is your appetite?", "options": {"a": "Normal", "b": "Slightly reduced", "c": "Poor", "d": "No appetite or overeating"}},
        {"question": "7. How often do you feel hopeless or down?", "options": {"a": "Never", "b": "Occasionally", "c": "Frequently", "d": "Almost always"}},
        {"question": "8. Are you able to concentrate on tasks?", "options": {"a": "Always", "b": "Usually", "c": "Sometimes", "d": "Rarely"}},
        {"question": "9. How often do you feel tired or low on energy?", "options": {"a": "Rarely", "b": "Sometimes", "c": "Often", "d": "Every day"}},
        {"question": "10. How often do you feel like you are not good enough?", "options": {"a": "Never", "b": "Rarely", "c": "Often", "d": "Almost always"}},
        {"question": "11. Do you often feel irritated or angry?", "options": {"a": "No", "b": "Occasionally", "c": "Often", "d": "Most of the time"}},
        {"question": "12. How well do you manage stress?", "options": {"a": "Very well", "b": "Moderately", "c": "Not well", "d": "Poorly"}},
        {"question": "13. Do you experience mood swings?", "options": {"a": "Rarely", "b": "Sometimes", "c": "Often", "d": "Very frequently"}},
        {"question": "14. How often do you feel lonely?", "options": {"a": "Never", "b": "Sometimes", "c": "Often", "d": "Almost always"}},
        {"question": "15. Do you worry about the future constantly?", "options": {"a": "No", "b": "Occasionally", "c": "Frequently", "d": "Always"}},
        {"question": "16. Do you avoid social situations?", "options": {"a": "Never", "b": "Sometimes", "c": "Often", "d": "Almost always"}},
        {"question": "17. Do you have trouble making decisions?", "options": {"a": "No", "b": "Occasionally", "c": "Often", "d": "Always"}},
        {"question": "18. Do you feel mentally exhausted?", "options": {"a": "Rarely", "b": "Sometimes", "c": "Often", "d": "Almost always"}},
        {"question": "19. Do you feel like a burden to others?", "options": {"a": "Never", "b": "Sometimes", "c": "Often", "d": "Always"}},
        {"question": "20. Are you finding it hard to stay motivated?", "options": {"a": "No", "b": "A little", "c": "Quite a bit", "d": "Yes, completely"}}
    ]

    scores = {"a": 1, "b": 2, "c": 3, "d": 4}
    total_score = 0

    for q in questions:
        print("\n" + q["question"])
        for key, value in q["options"].items():
            print(f"  {key}) {value}")
        while True:
            answer = input("Your answer (a/b/c/d): ").lower()
            if answer in scores:
                total_score += scores[answer]
                break
            else:
                print("❌ Invalid input. Please choose a, b, c, or d.")

    print("\n📝 Your Total Score:", total_score)


    if total_score <= 29:
        print("✅ You appear to be doing quite well. Keep maintaining your mental health!")
    elif total_score <= 49:
        print("⚠️ You might be experiencing mild to moderate emotional challenges. Consider self-care and support.")
    elif total_score <= 69:
        print("❗ You're facing significant stress. It's a good idea to talk to someone or seek professional support.")
    else:
        print("🚨 You may be going through serious mental health struggles. Please reach out to a mental health professional as soon as possible.")



# Run the quiz when the script is executed directly
if __name__ == "__main__":
    mental_health_quiz()

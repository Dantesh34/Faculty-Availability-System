"""
Faculty Availability Prediction Model
IoT-Based Occupancy Analytics System
Generates synthetic data, trains a Random Forest model,
and exports predictions + history to JSON for the dashboard.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Faculty Profiles ────────────────────────────────────────────────────────

FACULTY = [
    {
        "id": "F001", "name": "Dr. Anil Sharma", "dept": "Computer Science",
        "cabin": "CS-201",
        "arrive_mean": 9.0,  "arrive_std": 0.4,
        "depart_mean": 17.5, "depart_std": 0.5,
        "lunch_start": 13.0, "lunch_dur": 1.0,
        "absent_prob": 0.08,
        "late_days": [4],        # Friday often late
        "early_leave_days": [],
        "meeting_slots": [(11.0, 12.0, 3), (15.0, 16.0, 4)],  # (start, end, weekday)
    },
    {
        "id": "F002", "name": "Prof. Meena Iyer", "dept": "Electronics",
        "cabin": "EC-115",
        "arrive_mean": 9.5,  "arrive_std": 0.3,
        "depart_mean": 17.0, "depart_std": 0.4,
        "lunch_start": 13.5, "lunch_dur": 0.75,
        "absent_prob": 0.10,
        "late_days": [0],        # Monday late
        "early_leave_days": [4],
        "meeting_slots": [(10.0, 11.0, 2)],
    },
    {
        "id": "F003", "name": "Dr. Rajesh Kumar", "dept": "Mechanical",
        "cabin": "ME-302",
        "arrive_mean": 8.5,  "arrive_std": 0.5,
        "depart_mean": 16.5, "depart_std": 0.6,
        "lunch_start": 12.5, "lunch_dur": 1.25,
        "absent_prob": 0.06,
        "late_days": [],
        "early_leave_days": [2],  # Wednesday
        "meeting_slots": [(9.0, 10.0, 1), (14.0, 15.0, 3)],
    },
    {
        "id": "F004", "name": "Dr. Priya Nair", "dept": "Civil",
        "cabin": "CV-104",
        "arrive_mean": 9.2,  "arrive_std": 0.35,
        "depart_mean": 17.2, "depart_std": 0.45,
        "lunch_start": 13.0, "lunch_dur": 0.9,
        "absent_prob": 0.12,
        "late_days": [1, 4],
        "early_leave_days": [],
        "meeting_slots": [(14.0, 15.5, 0), (10.5, 11.5, 2)],
    },
    {
        "id": "F005", "name": "Prof. Suresh Babu", "dept": "Mathematics",
        "cabin": "MA-210",
        "arrive_mean": 8.8,  "arrive_std": 0.3,
        "depart_mean": 16.8, "depart_std": 0.4,
        "lunch_start": 13.2, "lunch_dur": 0.8,
        "absent_prob": 0.07,
        "late_days": [],
        "early_leave_days": [3],  # Thursday
        "meeting_slots": [(11.5, 12.5, 1), (15.5, 16.5, 3)],
    },
]

# ─── Synthetic Data Generator ────────────────────────────────────────────────

def generate_faculty_logs(faculty, start_date, end_date):
    """Generate minute-by-minute presence logs for one faculty member."""
    records = []
    current = start_date

    while current <= end_date:
        weekday = current.weekday()
        if weekday >= 5:  # Skip weekends
            current += timedelta(days=1)
            continue

        # Random absence
        if np.random.random() < faculty["absent_prob"]:
            current += timedelta(days=1)
            continue

        # Arrival time
        arrive = faculty["arrive_mean"] + np.random.normal(0, faculty["arrive_std"])
        if weekday in faculty["late_days"]:
            arrive += np.random.uniform(0.3, 0.8)

        # Departure time
        depart = faculty["depart_mean"] + np.random.normal(0, faculty["depart_std"])
        if weekday in faculty["early_leave_days"]:
            depart -= np.random.uniform(0.5, 1.5)

        arrive = max(8.0, min(arrive, 11.0))
        depart = max(14.0, min(depart, 19.0))

        # Build hourly presence slots for the day
        # Start with full presence
        absent_slots = []

        # Lunch break
        lunch_start = faculty["lunch_start"] + np.random.normal(0, 0.15)
        lunch_end = lunch_start + faculty["lunch_dur"] + np.random.normal(0, 0.1)
        absent_slots.append((lunch_start, lunch_end))

        # Meeting slots
        for (ms, me, mday) in faculty["meeting_slots"]:
            if mday == weekday and np.random.random() < 0.7:
                absent_slots.append((ms, me))

        # Generate 30-minute interval records
        hour = arrive
        while hour <= depart:
            in_cabin = True
            for (ab_s, ab_e) in absent_slots:
                if ab_s <= hour < ab_e:
                    in_cabin = False
                    break

            records.append({
                "faculty_id": faculty["id"],
                "faculty_name": faculty["name"],
                "dept": faculty["dept"],
                "cabin": faculty["cabin"],
                "date": current.strftime("%Y-%m-%d"),
                "weekday": weekday,
                "weekday_name": current.strftime("%A"),
                "hour": int(hour),
                "minute": int((hour % 1) * 60 // 30) * 30,
                "time_slot": round(hour * 2) / 2,  # 30-min slots
                "present": int(in_cabin),
            })
            hour += 0.5

        current += timedelta(days=1)

    return records


def generate_all_data():
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=180)

    all_records = []
    for f in FACULTY:
        print(f"  Generating data for {f['name']}...")
        records = generate_faculty_logs(f, start_date, end_date)
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    print(f"  Total records generated: {len(df):,}")
    return df


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_features(df):
    df = df.copy()
    df["time_slot_norm"] = df["time_slot"] / 24.0
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 5)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 5)
    df["hour_sin"] = np.sin(2 * np.pi * df["time_slot"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["time_slot"] / 24)
    df["is_lunch_zone"] = ((df["time_slot"] >= 12.5) & (df["time_slot"] <= 14.5)).astype(int)
    df["is_morning"] = (df["time_slot"] < 11.0).astype(int)
    df["is_afternoon"] = (df["time_slot"] >= 14.0).astype(int)

    # Historical average presence per faculty per slot
    avg = df.groupby(["faculty_id", "weekday", "time_slot"])["present"].mean().reset_index()
    avg.rename(columns={"present": "hist_avg"}, inplace=True)
    df = df.merge(avg, on=["faculty_id", "weekday", "time_slot"], how="left")
    df["hist_avg"] = df["hist_avg"].fillna(0.5)

    return df


FEATURE_COLS = [
    "time_slot_norm", "weekday_sin", "weekday_cos",
    "hour_sin", "hour_cos", "is_lunch_zone",
    "is_morning", "is_afternoon", "hist_avg"
]


# ─── Train Model ─────────────────────────────────────────────────────────────

def train_models(df):
    models = {}
    metrics = {}

    for faculty in FACULTY:
        fid = faculty["id"]
        fdf = df[df["faculty_id"] == fid].copy()
        fdf = build_features(fdf)

        X = fdf[FEATURE_COLS]
        y = fdf["present"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        models[fid] = model
        metrics[fid] = {
            "accuracy": round(acc * 100, 1),
            "name": faculty["name"],
        }
        print(f"  {faculty['name']}: accuracy = {acc*100:.1f}%")

    return models, metrics


# ─── Generate Predictions ────────────────────────────────────────────────────

def generate_predictions(models, df):
    """Generate predictions for the next 7 days."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    predictions = []

    # Build historical averages from training data
    hist_df = build_features(df)
    hist_avg_map = hist_df.groupby(["faculty_id", "weekday", "time_slot"])["present"].mean().to_dict()

    for day_offset in range(14):
        date = today + timedelta(days=day_offset)
        weekday = date.weekday()
        if weekday >= 5:
            continue

        for faculty in FACULTY:
            fid = faculty["id"]
            model = models[fid]

            for slot in [s / 2 for s in range(17, 36)]:  # 8:30 to 18:00
                hist_avg = hist_avg_map.get((fid, weekday, slot), 0.5)

                row = pd.DataFrame([{
                    "time_slot_norm": slot / 24.0,
                    "weekday_sin": np.sin(2 * np.pi * weekday / 5),
                    "weekday_cos": np.cos(2 * np.pi * weekday / 5),
                    "hour_sin": np.sin(2 * np.pi * slot / 24),
                    "hour_cos": np.cos(2 * np.pi * slot / 24),
                    "is_lunch_zone": int(12.5 <= slot <= 14.5),
                    "is_morning": int(slot < 11.0),
                    "is_afternoon": int(slot >= 14.0),
                    "hist_avg": hist_avg,
                }])

                proba = model.predict_proba(row)[0][1]
                predicted = int(proba >= 0.5)

                predictions.append({
                    "faculty_id": fid,
                    "faculty_name": faculty["name"],
                    "dept": faculty["dept"],
                    "cabin": faculty["cabin"],
                    "date": date.strftime("%Y-%m-%d"),
                    "weekday": weekday,
                    "weekday_name": date.strftime("%A"),
                    "time_slot": slot,
                    "time_label": f"{int(slot):02d}:{int((slot%1)*60):02d}",
                    "probability": round(proba, 3),
                    "predicted_present": predicted,
                })

    return predictions


# ─── Today's Live Status ──────────────────────────────────────────────────────

def get_current_status(models, df):
    """Simulate current live status for all faculty."""
    now = datetime.now()
    current_slot = round(now.hour + now.minute / 60, 1)
    current_slot = round(current_slot * 2) / 2  # nearest 30-min
    weekday = now.weekday()

    hist_df = build_features(df)
    hist_avg_map = hist_df.groupby(["faculty_id", "weekday", "time_slot"])["present"].mean().to_dict()

    statuses = []
    for faculty in FACULTY:
        fid = faculty["id"]
        model = models[fid]

        hist_avg = hist_avg_map.get((fid, weekday, current_slot), 0.5)

        row = pd.DataFrame([{
            "time_slot_norm": current_slot / 24.0,
            "weekday_sin": np.sin(2 * np.pi * weekday / 5),
            "weekday_cos": np.cos(2 * np.pi * weekday / 5),
            "hour_sin": np.sin(2 * np.pi * current_slot / 24),
            "hour_cos": np.cos(2 * np.pi * current_slot / 24),
            "is_lunch_zone": int(12.5 <= current_slot <= 14.5),
            "is_morning": int(current_slot < 11.0),
            "is_afternoon": int(current_slot >= 14.0),
            "hist_avg": hist_avg,
        }])

        proba = model.predict_proba(row)[0][1]
        present = proba >= 0.5

        # Next available slot prediction
        next_avail = None
        if not present:
            for future_slot in [current_slot + s * 0.5 for s in range(1, 12)]:
                if future_slot > 18:
                    break
                ha = hist_avg_map.get((fid, weekday, future_slot), 0.5)
                r = pd.DataFrame([{
                    "time_slot_norm": future_slot / 24.0,
                    "weekday_sin": np.sin(2 * np.pi * weekday / 5),
                    "weekday_cos": np.cos(2 * np.pi * weekday / 5),
                    "hour_sin": np.sin(2 * np.pi * future_slot / 24),
                    "hour_cos": np.cos(2 * np.pi * future_slot / 24),
                    "is_lunch_zone": int(12.5 <= future_slot <= 14.5),
                    "is_morning": int(future_slot < 11.0),
                    "is_afternoon": int(future_slot >= 14.0),
                    "hist_avg": ha,
                }])
                p = model.predict_proba(r)[0][1]
                if p >= 0.6:
                    h = int(future_slot)
                    m = int((future_slot % 1) * 60)
                    next_avail = f"{h:02d}:{m:02d}"
                    break

        statuses.append({
            "faculty_id": fid,
            "faculty_name": faculty["name"],
            "dept": faculty["dept"],
            "cabin": faculty["cabin"],
            "present": bool(present),
            "probability": round(float(proba), 3),
            "status": "Available" if present else "Away",
            "next_available": next_avail,
            "last_updated": now.strftime("%H:%M:%S"),
        })

    return statuses


# ─── Historical Summary ───────────────────────────────────────────────────────

def get_history_summary(df):
    """Weekly availability heatmap data per faculty."""
    summary = {}
    for faculty in FACULTY:
        fid = faculty["id"]
        fdf = df[df["faculty_id"] == fid]
        pivot = fdf.groupby(["weekday", "time_slot"])["present"].mean().reset_index()
        pivot["time_label"] = pivot["time_slot"].apply(lambda x: f"{int(x):02d}:{int((x%1)*60):02d}")
        pivot["weekday_name"] = pivot["weekday"].map({
            0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"
        })
        summary[fid] = {
            "heatmap": pivot[["weekday_name", "time_label", "present"]].to_dict(orient="records"),
            "avg_daily_hours": round(fdf.groupby("date")["present"].sum().mean() * 0.5, 1),
            "attendance_rate": round(fdf["present"].mean() * 100, 1),
        }
    return summary


# ─── Main Export ─────────────────────────────────────────────────────────────

def main():
    print("\n🔵 Faculty Availability Prediction System")
    print("=" * 50)

    print("\n[1/4] Generating synthetic sensor data...")
    df = generate_all_data()

    print("\n[2/4] Training Random Forest models...")
    models, metrics = train_models(df)

    print("\n[3/4] Generating predictions & live status...")
    predictions = generate_predictions(models, df)
    current_status = get_current_status(models, df)
    history = get_history_summary(df)

    print("\n[4/4] Exporting to JSON...")

    output = {
        "generated_at": datetime.now().isoformat(),
        "model_metrics": metrics,
        "current_status": current_status,
        "predictions": predictions,
        "history": history,
        "faculty_list": [
            {"id": f["id"], "name": f["name"], "dept": f["dept"], "cabin": f["cabin"]}
            for f in FACULTY
        ],
    }

    with open("/home/claude/faculty_data.json", "w") as fp:
        json.dump(output, fp, indent=2)

    print("  Saved: faculty_data.json")
    print("\n✅ Model training complete!")
    print("\nModel Accuracies:")
    for fid, m in metrics.items():
        print(f"  {m['name']}: {m['accuracy']}%")

    return output


if __name__ == "__main__":
    main()

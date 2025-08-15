from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pandas as pd
import re
from openai import OpenAI
from dotenv import load_dotenv
from flask_sqlalchemy import SQLAlchemy

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Database
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize DB object
db = SQLAlchemy(app)

# Config
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"xlsx"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple user store (replace with DB in production)
USERS = {
    "tda_s_user": {"password": "pass123", "role": "TDA-S"},
    "tda_m_user": {"password": "pass123", "role": "TDA-M"}
}

# ---------- Helpers ----------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_goal_text(goal):
    """Remove square bracket content and leading number from goal string."""
    if pd.isna(goal):
        return ""
    text = str(goal)
    # Remove [ ... ] part
    text = re.sub(r"\s*\[.*?\]", "", text)
    # Remove leading number + dot + space (e.g., "3. " or "19. ")
    text = re.sub(r"^\d+\.\s*", "", text)
    return text.strip()

# ---------- Routes ----------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user = USERS.get(username)

    if user and user["password"] == password:
        return jsonify({"success": True, "role": user["role"]})
    return jsonify({"success": False, "message": "Invalid credentials"}), 401


@app.route("/upload", methods=["POST"])
def upload_file():
    role = request.form.get("role")
    if role != "TDA-S":
        return jsonify({"error": "Unauthorized"}), 403

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)
        return jsonify({"success": True, "filename": filename})
    else:
        return jsonify({"error": "Invalid file type"}), 400


@app.route("/files", methods=["GET"])
def list_files():
    role = request.args.get("role")
    if role != "TDA-M":
        return jsonify({"error": "Unauthorized"}), 403

    files = os.listdir(UPLOAD_FOLDER)
    return jsonify(files)


def extract_domain(goal):
    """Extract the domain name from the [Domain: ...] part of the goal string."""
    if pd.isna(goal):
        return ""
    match = re.search(r"\[Domain:\s*([^,\]]+)", str(goal), re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""

@app.route("/goals/<filename>", methods=["GET"])
def get_goals(filename):
    role = request.args.get("role")
    search_query = request.args.get("search", "").strip().lower()

    if role != "TDA-M":
        return jsonify({"error": "Unauthorized"}), 403

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    try:
        # Read the Goals sheet
        df = pd.read_excel(file_path, sheet_name="Goals")

        # Normalize headers to lowercase, strip spaces, collapse multiple spaces
        df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]

        goal_col = "goal"
        attain_col = "attainment status"
        steps_col = "steps"
        comments_col = "comments"

        # Check required columns
        for col in [goal_col, attain_col, steps_col, comments_col]:
            if col not in df.columns:
                return jsonify({
                    "error": f"Required column '{col}' not found",
                    "found_columns": df.columns.tolist()
                }), 400

        results = []
        for _, row in df.iterrows():
            original_goal = row[goal_col]
            goal_cleaned = clean_goal_text(original_goal)
            domain = extract_domain(original_goal)
            attainment_status = row[attain_col] if pd.notna(row[attain_col]) else ""
            steps = row[steps_col] if pd.notna(row[steps_col]) else ""
            comments = row[comments_col] if pd.notna(row[comments_col]) else ""

            if search_query:
                if search_query in goal_cleaned.lower():
                    results.append({
                        "goal": goal_cleaned,
                        "attainment_status": attainment_status,
                        "domain": domain,
                        "steps": steps,
                        "comments": comments
                    })
            else:
                results.append({
                    "goal": goal_cleaned,
                    "attainment_status": attainment_status,
                    "domain": domain,
                    "steps": steps,
                    "comments": comments
                })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

import traceback

@app.route("/sheet/<filename>/<sheetname>", methods=["GET"])
def get_sheet(filename, sheetname):
    role = request.args.get("role")
    search_query = request.args.get("search", "").strip().lower()

    print(f"[DEBUG] Request to /sheet → file={filename}, sheet={sheetname}, role={role}, search='{search_query}'")

    if role != "TDA-M":
        print("[DEBUG] Unauthorized access attempt")
        return jsonify({"error": "Unauthorized"}), 403

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        print(f"[DEBUG] File not found: {file_path}")
        return jsonify({"error": "File not found"}), 404

    try:
        # ---------------- GOALS ----------------
        if sheetname.lower() == "goals":
            df = pd.read_excel(file_path, sheet_name=sheetname)
            print(f"[DEBUG] Raw columns: {df.columns.tolist()}")
            df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]
            print(f"[DEBUG] Normalized columns: {df.columns.tolist()}")

            required_cols = [
                "goal", "attainment status", "steps", "comments",
                "keys", "justification", "hbi", "rbi", "commbi", "ii", "settings", "percentage"
            ]
            for col in required_cols:
                if col not in df.columns:
                    return jsonify({"error": f"Missing column {col}", "found_columns": df.columns.tolist()}), 400

            results = []
            for idx, row in df.iterrows():
                goal_val = row.get("goal", "")
                if pd.isna(goal_val):
                    continue
                goal_cleaned = clean_goal_text(goal_val)
                domain = extract_domain(goal_val)
                if search_query and search_query not in goal_cleaned.lower():
                    continue
                results.append({
                    "goal": goal_cleaned,
                    "attainment_status": row["attainment status"] if pd.notna(row["attainment status"]) else "",
                    "domain": domain,
                    "steps": row["steps"] if pd.notna(row["steps"]) else "",
                    "comments": row["comments"] if pd.notna(row["comments"]) else "",
                    "keys": row["keys"] if pd.notna(row["keys"]) else "",
                    "justification": row["justification"] if pd.notna(row["justification"]) else "",
                    "hbi": row["hbi"] if pd.notna(row["hbi"]) else "",
                    "rbi": row["rbi"] if pd.notna(row["rbi"]) else "",
                    "commbi": row["commbi"] if pd.notna(row["commbi"]) else "",
                    "ii": row["ii"] if pd.notna(row["ii"]) else "",
                    "settings": row["settings"] if pd.notna(row["settings"]) else "",
                    "percentage": row["percentage"] if pd.notna(row["percentage"]) else ""
                })
            return jsonify(results)

        # ---------------- ASSESSMENT SHEET ----------------
        elif sheetname.lower() == "assessment sheet":
            df = pd.read_excel(file_path, sheet_name=sheetname)
            df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]
            required_cols = [
                "goal", "steps", "comments",
                "keys", "justification", "hbi", "rbi", "commbi", "ii", "settings", "percentage"
            ]
            for col in required_cols:
                if col not in df.columns:
                    return jsonify({"error": f"Missing column {col}", "found_columns": df.columns.tolist()}), 400

            results = []
            for idx, row in df.iterrows():
                goal_val = row.get("goal", "")
                if pd.isna(goal_val):
                    continue
                goal_cleaned = clean_goal_text(goal_val)
                domain = extract_domain(goal_val)
                if search_query and search_query not in goal_cleaned.lower():
                    continue
                results.append({
                    "goal": goal_cleaned,
                    "domain": domain,
                    "steps": row["steps"] if pd.notna(row["steps"]) else "",
                    "comments": row["comments"] if pd.notna(row["comments"]) else "",
                    "keys": row["keys"] if pd.notna(row["keys"]) else "",
                    "justification": row["justification"] if pd.notna(row["justification"]) else "",
                    "hbi": row["hbi"] if pd.notna(row["hbi"]) else "",
                    "rbi": row["rbi"] if pd.notna(row["rbi"]) else "",
                    "commbi": row["commbi"] if pd.notna(row["commbi"]) else "",
                    "ii": row["ii"] if pd.notna(row["ii"]) else "",
                    "settings": row["settings"] if pd.notna(row["settings"]) else "",
                    "percentage": row["percentage"] if pd.notna(row["percentage"]) else ""
                })
            return jsonify(results)

        # ---------------- RECOMMENDATIONS ----------------
        elif sheetname.lower() == "recommendations":
            df = pd.read_excel(file_path, sheet_name=sheetname)
            df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]
            required_cols = ["recommendation", "result"]
            for col in required_cols:
                if col not in df.columns:
                    return jsonify({"error": f"Missing column {col}", "found_columns": df.columns.tolist()}), 400
            results = [
                {"recommendation": row["recommendation"] if pd.notna(row["recommendation"]) else "",
                 "result": row["result"] if pd.notna(row["result"]) else ""}
                for _, row in df.iterrows()
            ]
            return jsonify(results)

        # ---------------- TRANSMISSION NOTES ----------------
        elif sheetname.lower() == "transmission notes":
            df = pd.read_excel(file_path, sheet_name=sheetname)
            df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]
            required_cols = ["transmission notes", "results"]
            for col in required_cols:
                if col not in df.columns:
                    return jsonify({"error": f"Missing column {col}", "found_columns": df.columns.tolist()}), 400
            results = [
                {"transmission_notes": row["transmission notes"] if pd.notna(row["transmission notes"]) else "",
                 "results": row["results"] if pd.notna(row["results"]) else ""}
                for _, row in df.iterrows()
            ]
            return jsonify(results)

        # ---------------- ADDITIONAL NOTES ----------------
        elif sheetname.lower() == "additional notes":
            print(f"[DEBUG] Fetching Additional Notes from 'Student Details' sheet")
            df = pd.read_excel(file_path, sheet_name="Student Details")
            df.columns = [re.sub(r'\s+', ' ', c).strip().lower() for c in df.columns]
            if "notes" not in df.columns:
                return jsonify({"error": "Missing 'Notes' column"}), 400

            first_note = ""
            if not df.empty:
                raw_val = df.iloc[0]["notes"]
                if pd.notna(raw_val) and str(raw_val).strip().upper() != "FILL":
                    first_note = str(raw_val).strip()

            return jsonify({"notes": first_note})

        else:
            return jsonify({"error": "Unknown sheet requested"}), 400

    except Exception as e:
        print(f"[ERROR] Exception while reading sheet '{sheetname}' from file '{filename}': {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate-suggestions", methods=["POST"])
def generate_suggestions():
    try:
        data = request.json
        goals = data.get("goals", [])

        # Only keep partially attained goals
        partially_attained = [
            {
                "goalName": g.get("goal", "").strip(),
                "comments": g.get("comments", "").strip()
            }
            for g in goals
            if g.get("attainment_status", "").strip().lower() == "partially attained"
        ]

        if not partially_attained:
            return jsonify({"toBeSelectedAgain": []})

        # Build AI prompt
        prompt = f"""
You are an AI that analyzes educational goals and determines which ones
should be selected again for the next cycle.

- Input: A list of goals with their comments.
- Output: ONLY valid JSON in this format:

{{
  "toBeSelectedAgain": [
    {{ "goalName": "string", "reason": "string" }}
  ]
}}

- "reason" must clearly mention relevant parts of the provided comments.
- Include ALL goals that should be repeated, based on comments or context.
Here is the input:
{partially_attained}
"""

        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=800,
            messages=[
                {"role": "system", "content": "You output only JSON. No markdown, no extra text."},
                {"role": "user", "content": prompt}
            ]
        )

        raw_output = response.choices[0].message.content.strip()

        # Extract JSON
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}")
        if json_start == -1 or json_end == -1:
            return jsonify({"error": "Invalid AI output", "raw_output": raw_output}), 400

        parsed = raw_output[json_start:json_end+1]
        return jsonify(eval(parsed))  # safe here since AI returns JSON format

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

# -*- coding: utf-8 -*-
import os
import io
import json
import base64
import sqlite3
import joblib
import numpy as np
import cv2
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename

# TensorFlow
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Gemini Vision LLM
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ====================================================
#                U-NET SEGMENTATION MODEL
# ====================================================

UNET_MODEL_PATH = "best_unet_checkpoint.keras"
UNET_IMG_SIZE = (256, 256)

# Custom Dice Loss (must match training)
def dice_loss(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    smooth = 1e-6
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )
    return 1 - dice

# Load U-Net Model
try:
    unet_model = load_model(
        UNET_MODEL_PATH,
        custom_objects={"dice_loss": dice_loss},
        compile=False
    )
    print("U-Net Model Loaded Successfully.")
except Exception as e:
    print("ERROR loading U-Net model:", e)
    unet_model = None


def unet_predict_mask(image_bytes):
    """
    Takes raw uploaded image bytes and returns (mask_b64, overlay_b64)
    """

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    orig_h, orig_w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(img_rgb, UNET_IMG_SIZE)

    x = resized.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    pred = unet_model.predict(x)[0]
    mask = (pred > 0.5).astype(np.uint8)

    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    mask_out = (mask_resized * 255).astype(np.uint8)

    overlay = img.copy()
    overlay[mask_out == 255] = (0, 255, 0)
    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    _, mask_png = cv2.imencode(".png", mask_out)
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    _, overlay_png = cv2.imencode(".png", blended)
    overlay_b64 = base64.b64encode(overlay_png).decode("utf-8")

    return mask_b64, overlay_b64


# ====================================================
#                 GEMINI CLIENT (OCR + Q&A)
# ====================================================

GEMINI_MODEL = "gemini-2.5-flash"
client = None
try:
    client = genai.Client()
    print("Gemini client initialized.")
except Exception as e:
    print("Gemini init error:", e)


# ====================================================
#               XGBOOST PCOS MODEL
# ====================================================

try:
    xgb_model = joblib.load("pcos_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("PCOS ML Model Loaded.")
except Exception as e:
    print("Failed to load XGB/Scaler:", e)
    xgb_model = None
    scaler = None


# ====================================================
#                    FLASK APP
# ====================================================

app = Flask(__name__)
app.secret_key = "super_secret_key"

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ====================================================
#                DATABASE INITIALIZATION
# ====================================================

def init_db():
    if not os.path.exists("pcos.db"):
        conn = sqlite3.connect("pcos.db")
        c = conn.cursor()

        c.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT,
                role TEXT
            )
        """)

        c.execute("""
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle TEXT,
                weight REAL,
                acne TEXT,
                risk TEXT
            )
        """)

        c.execute("INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
                  ("Admin", "admin@hospital.com", "admin123", "admin"))

        conn.commit()
        conn.close()
        print("Database created.")

init_db()


def get_db_connection():
    conn = sqlite3.connect("pcos.db")
    conn.row_factory = sqlite3.Row
    return conn


# ====================================================
#                    FLASK ROUTES
# ====================================================

@app.route("/")
def patient_home():
    return render_template("patient_home.html")


# ========================== U-NET ROUTE ==========================

@app.route("/unet_predict", methods=["POST"])
def unet_predict_route():
    if unet_model is None:
        return jsonify({"error": "U-Net model failed to load."})

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."})

    try:
        image_bytes = file.read()
        mask_b64, overlay_b64 = unet_predict_mask(image_bytes)

        return jsonify({
            "unet_mask": mask_b64,
            "unet_overlay": overlay_b64
        })

    except Exception as e:
        return jsonify({"error": f"Segmentation Error: {str(e)}"})



# ====================================================
#               LOGIN AND DASHBOARD ROUTES
# ====================================================

@app.route("/hcw")
def hcw_home():
    return render_template("hcw_home.html")

@app.route("/hcw/login", methods=["POST"])
def hcw_login():
    email = request.form["email"]
    password = request.form["password"]

    conn = get_db_connection()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email, password)
    ).fetchone()
    conn.close()

    if user:
        session["user_id"] = user["id"]
        session["role"] = user["role"]
        session["name"] = user["name"]

        if user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("doctor_dashboard"))
    else:
        flash("Invalid login", "error")
        return redirect(url_for("hcw_home"))
    
@app.route("/doctor/ocr_process", methods=["POST"])
def doctor_ocr_process():
    try:
        file = request.files.get("report_image")
        if not file:
            return jsonify({"status": "error", "message": "No file uploaded"})

        image_bytes = file.read()

        # Gemini Vision OCR
        result = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    parts=[types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")]
                )
            ]
        )

        extracted_text = result.text

        # You must improve this mapping for your own lab reports
        extracted_data = {
            "Height": "",
            "Weight": "",
            "FSH": "",
            "LH": "",
            "Waist": "",
            "Hip": ""
        }

        return jsonify({"status": "success", "data": extracted_data})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    


@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("hcw_home"))

    conn = get_db_connection()
    doctors = conn.execute("SELECT * FROM users WHERE role='doctor'").fetchall()
    conn.close()
    return render_template("admin_dashboard.html", doctors=doctors)


@app.route("/doctor")
def doctor_dashboard():
    if session.get("role") != "doctor":
        return redirect(url_for("hcw_home"))

    conn = get_db_connection()
    patients = conn.execute("SELECT * FROM patients").fetchall()
    conn.close()
    return render_template("doctor_dashboard.html", patients=patients)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("hcw_home"))



# ====================================================
#                    MAIN
# ====================================================

if __name__ == "__main__":
    app.run(debug=True)

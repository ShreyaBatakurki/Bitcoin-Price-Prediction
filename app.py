#from flask import Flask, render_template, request, redirect, url_for, session
#from flask import Flask, render_template, request, redirect, url_for, session, flash

import re
from distutils.log import debug
from os import environ
from flask import *
import os
from flask_cors import CORS
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import hashlib


import warnings
warnings.filterwarnings("ignore")

# --- add near the top with your other imports ---
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from flask import jsonify
# -----------------------------------------------

# =========== ML: helper to fetch BTC history ===========
def get_btc_history_usd(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull daily BTC-USD prices from CoinGecko in [start_date, end_date] (YYYY-MM-DD).
    Returns DataFrame with columns: date (datetime.date), close (float).
    """
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end   = datetime.strptime(end_date, "%Y-%m-%d")
    # add buffer (30 days) for lag features
    dt_buf   = dt_start - timedelta(days=300)

    ts_from = int(time.mktime(dt_buf.timetuple()))
    ts_to   = int(time.mktime((dt_end + timedelta(days=1)).timetuple()))  # inclusive

    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        f"?vs_currency=usd&from={ts_from}&to={ts_to}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    # 'prices' is list of [ms_timestamp, price]
    prices = data.get("prices", [])
    if not prices:
        raise ValueError("No price data from CoinGecko.")

    df = pd.DataFrame(prices, columns=["ts_ms", "close"])
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.date
    # daily aggregate (CoinGecko may return multiple intraday points)
    df = df.groupby("date", as_index=False)["close"].mean()

    # trim strictly to requested range (but lags need buffer internally)
    return df

def build_lag_features(df: pd.DataFrame, lags=14) -> pd.DataFrame:
    out = df.copy()
    for L in range(1, lags + 1):
        out[f"lag_{L}"] = out["close"].shift(L)
    out = out.dropna().reset_index(drop=True)
    return out

def iterative_predict(model, last_known_prices: list, horizon_days: int) -> list:
    """
    last_known_prices: list of most recent prices ordered oldest->newest (length = lags)
    horizon_days: how many future days to predict iteratively
    """
    preds = []
    window = last_known_prices.copy()
    for _ in range(horizon_days):
        X = np.array(window[::-1])  # last_known with lag_1 as most recent; we'll match training order below
        X = X.reshape(1, -1)
        yhat = model.predict(X)[0]
        preds.append(float(yhat))
        # advance window
        window.pop(0)
        window.append(yhat)
    return preds

# Global variables to store model and evaluation metrics
model_global = None
model_metrics = {}

# =========== API: Predict BTC prices ===========

app = Flask(__name__)
CORS(app)
app.secret_key = "abc"


# MySQL Config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'bitcoinprice'
mysql = MySQL(app)

# Regex Patterns
name_pattern = re.compile(r'^[A-Za-z]+ [A-Za-z]+$')
email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
mobile_pattern = re.compile(r'^[6-9]\d{9}$')
password_pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$')



# Dummy user store
users = {"admin": "password"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

from flask import jsonify

@app.route("/signup", methods=["POST"])
def signup():
    name = request.form.get("name")
    email = request.form.get("email")
    mobile = request.form.get("mobile")
    password = request.form.get("password")

    # Validation
    if not name_pattern.match(name):
        return jsonify({"status": "error", "message": "Enter full name (First and Last, only letters)."})
    if not email_pattern.match(email):
        return jsonify({"status": "error", "message": "Invalid Email format."})
    if not mobile_pattern.match(mobile):
        return jsonify({"status": "error", "message": "Invalid Mobile number (10 digits starting with 6-9)."})
    if not password_pattern.match(password):
        return jsonify({"status": "error", "message": "Password must be 8+ chars with Upper, Lower, Number & Special char."})

    # Check existing user
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email=%s OR mobile=%s", (email, mobile))
    if cur.fetchone():
        cur.close()
        return jsonify({"status": "error", "message": "Email or Mobile already registered."})

    # Insert new user
    hashed_password = generate_password_hash(password)
    cur.execute("INSERT INTO users (name, email, mobile, password) VALUES (%s, %s, %s, %s)",
                (name, email, mobile, hashed_password))
    mysql.connection.commit()
    cur.close()

    return jsonify({"status": "success", "message": "Registration successful! Please log in."})


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email")
    password = request.form.get("password")

    cur = mysql.connection.cursor()
    cur.execute("SELECT id, name, email, mobile, password FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    cur.close()

    if user and check_password_hash(user[4], password):
        session["user"] = {"id": user[0], "name": user[1], "email": user[2], "mobile": user[3]}
        return jsonify({"status": "success", "message": "Login successful!"})
    else:
        return jsonify({"status": "error", "message": "Invalid credentials."})


@app.route("/update_password", methods=["POST"])
def update_password():
    if "user" not in session:
        return jsonify(status="error", message="Session expired. Please log in again.")

    old_pass = request.form.get("old_password")
    new_pass = request.form.get("new_password")

    cur = mysql.connection.cursor()
    cur.execute("SELECT password FROM users WHERE id=%s", (session["user"]["id"],))
    current = cur.fetchone()

    if current and check_password_hash(current[0], old_pass):
        if not password_pattern.match(new_pass):
            return jsonify(status="error", message="New password must have 8+ chars, Upper, Lower, Number & Special char.")

        hashed = generate_password_hash(new_pass)
        cur.execute("UPDATE users SET password=%s WHERE id=%s", (hashed, session["user"]["id"]))
        mysql.connection.commit()
        cur.close()
        return jsonify(status="success", message="Password updated successfully!")
    else:
        return jsonify(status="error", message="Old password is incorrect.")

@app.route("/dashboard")
def dashboard():
    if "user" in session:
        return render_template("dashboard.html", user=session["user"])
    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("home"))


# =========== API: Predict BTC prices ===========
@app.route("/predict_prices", methods=["POST"])
def predict_prices():
    if "user" not in session:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    from_date = request.form.get("from_date")
    to_date   = request.form.get("to_date")

    try:
        # 1) fetch history with buffer
        df_all = get_btc_history_usd(from_date, to_date)

        # define lags & split
        LAGS = 14
        df_all_sorted = df_all.sort_values("date").reset_index(drop=True)

        # training set: all available dates <= day before prediction window starts
        dt_from = datetime.strptime(from_date, "%Y-%m-%d").date()
        train_df = df_all_sorted[df_all_sorted["date"] < dt_from].copy()

        # if not enough training days, widen to first 120 days minimum
        if len(train_df) < (LAGS + 30):
            # use as much as we can before from_date
            pass

        # 2) build lag features on training
        tr = build_lag_features(train_df[["date", "close"]], lags=LAGS)
        if tr.empty:
            return jsonify({"ok": False, "error": "Not enough history to train the model."}), 400

        # X order: lag_1 (yesterday), lag_2, ... lag_L
        lag_cols = [f"lag_{i}" for i in range(1, LAGS + 1)]
        X_train = tr[lag_cols].values
        y_train = tr["close"].values

        # Split data for training and testing
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=False
        )

        # 3) train model
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train_split, y_train_split)

        # Store model globally for future predictions
        global model_global, model_metrics
        model_global = model

        # 4) Evaluate model performance
        y_pred = model.predict(X_test_split)
        
        
        # Calculate accuracy as percentage within 5% tolerance
        accuracy_within_5pct = np.mean(np.abs((y_test_split - y_pred) / y_test_split) <= 0.05) * 100
        

        # Print model evaluation (one time at runtime)
        print("\n" + "="*50)
        print("MODEL Accuracy")
        print("="*50)
        print(f"Accuracy : {accuracy_within_5pct:.2f}%")
        print("="*50 + "\n")

        # 5) prepare iterative prediction start state
        # Take the last LAGS closes available up to (from_date - 1)
        hist_for_seed = df_all_sorted[df_all_sorted["date"] < dt_from]["close"].values
        if len(hist_for_seed) < LAGS:
            return jsonify({"ok": False, "error": "Insufficient history for lags."}), 400
        last_window = list(hist_for_seed[-LAGS:])  # oldest->newest

        # 6) generate date index
        dt_to = datetime.strptime(to_date, "%Y-%m-%d").date()
        future_days = (dt_to - dt_from).days + 1
        if future_days <= 0:
            return jsonify({"ok": False, "error": "Invalid date range."}), 400

        preds = iterative_predict(model, last_window, future_days)

        # 7) pack response with model metrics
        dates = [(dt_from + timedelta(days=i)).isoformat() for i in range(future_days)]
        rows = [{"date": d, "predicted": round(p, 2)} for d, p in zip(dates, preds)]
        
        response_data = {
            "ok": True, 
            "rows": rows,
            "accuracy": round(accuracy_within_5pct, 2)
        }
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# =========== API: Get Model Metrics ===========
@app.route("/model_metrics", methods=["GET"])
def get_model_metrics():
    if "user" not in session:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    
    if not model_metrics:
        return jsonify({"ok": False, "error": "No model trained yet"}), 404
    
    return jsonify({
        "ok": True,
        "metrics": model_metrics
    })

# =========== API: Real Block Validation ===========
@app.route("/validate_block", methods=["POST"])
def validate_block():
    if "user" not in session:
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    block_height = request.form.get("block_id", "").strip()
    input_hash   = request.form.get("hash_value", "").strip().lower()

    if not block_height.isdigit():
        return jsonify({"ok": False, "error": "Block height must be a number."}), 400

    try:
        # Get canonical hash for this height from Blockstream API
        # https://blockstream.info/api/block-height/{height}
        url = f"https://blockstream.info/api/block-height/{block_height}"
        r = requests.get(url, timeout=15)
        if r.status_code == 404:
            return jsonify({"ok": False, "error": "Block height not found."}), 404
        r.raise_for_status()
        expected_hash = r.text.strip().lower()

        valid = (expected_hash == input_hash)
        # (Optional) Pull block details if valid
        block_details = None
        if valid:
            d = requests.get(f"https://blockstream.info/api/block/{expected_hash}", timeout=20)
            if d.ok:
                j = d.json()
                block_details = {
                    "height": j.get("height"),
                    "timestamp": j.get("timestamp"),
                    "tx_count": j.get("tx_count"),
                    "size": j.get("size"),
                    "merkle_root": j.get("merkle_root")
                }

        return jsonify({
            "ok": True,
            "valid": valid,
            "expected_hash": expected_hash,
            "details": block_details
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---- Function to recompute block hash using Bitcoin's double SHA256 ----
def double_sha256(data_hex):
    first = hashlib.sha256(bytes.fromhex(data_hex)).digest()
    second = hashlib.sha256(first).digest()
    return second[::-1].hex()  # Bitcoin shows reversed


# ---- Get block data from blockchain API ----
def get_block_details(block_id):
    try:
        if block_id.isdigit():  # If Block Number
            url = f"https://blockchain.info/block-height/{block_id}?format=json"
            res = requests.get(url).json()
            block = res["blocks"][0]
        else:  # If Hash
            url = f"https://blockchain.info/rawblock/{block_id}"
            block = requests.get(url).json()
        return block
    except Exception as e:
        print("Error fetching block:", e)
        return None


@app.route("/hash_validate", methods=["POST"])
def hash_validate():
    block_id = request.form.get("block_query")  # matches HTML <input name="block_query">
    block = get_block_details(block_id)

    if not block:
        return jsonify({"ok": False, "error": "Block not found or API error."})

    # Build raw header for hashing
    try:
        version = int(block["ver"]).to_bytes(4, "little").hex()
        prev_block = bytes.fromhex(block["prev_block"])[::-1].hex()
        merkle_root = bytes.fromhex(block["mrkl_root"])[::-1].hex()
        timestamp = int(block["time"]).to_bytes(4, "little").hex()
        bits = int(block["bits"]).to_bytes(4, "little").hex()
        nonce = int(block["nonce"]).to_bytes(4, "little").hex()

        header_hex = version + prev_block + merkle_root + timestamp + bits + nonce
        recomputed_hash = double_sha256(header_hex)
        original_hash = block["hash"]

        valid = (recomputed_hash == original_hash)

        return jsonify({
            "ok": True,
            "valid": valid,
            "details": {
                "height": block.get("height"),
                "hash": original_hash,
                "timestamp": block.get("time"),
                "tx_count": block.get("n_tx"),
                "merkle_root": block.get("mrkl_root"),
            }
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route('/shutdown')
def shutdown():
    sys.exit()
    os.exit(0)
    return

'''
if __name__ == "__main__":
    app.run(debug=True)
'''

if __name__ == '__main__':
   HOST = environ.get('SERVER_HOST', '0.0.0.0')
   #HOST = environ.get('SERVER_HOST', 'localhost')
   try:
      PORT = int(environ.get('SERVER_PORT', '5555'))
   except ValueError:
      PORT = 5555
   app.run(HOST, PORT)
   #app.run(debug=True)

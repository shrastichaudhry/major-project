"""
AyurScan Backend — Complete & Final
Model  : EfficientNetB0 (.h5)
Plants : Aloe_Vera, Amla, Ashwagandha, Brahmi, Curry_Leaf,
         Giloy, Moringa, Neem, Tulsi, Turmeric  (10 classes)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import numpy as np
import os, io, base64, requests

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════
#  MODEL CONFIG — Google Drive auto-download
# ══════════════════════════════════════════════
MODEL_PATH     = "ayurscan_81percent_BEST.h5"
GDRIVE_FILE_ID = "1IsHzjDXYX97FPYNilR__hr9VzNoEAtDV"
# ══════════════════════════════════════════════

def download_model_from_drive():
    if os.path.exists(MODEL_PATH):
        print(f"   Model already exists: {MODEL_PATH}")
        return
    print("   Downloading model from Google Drive...")
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
    response = session.get(url, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={GDRIVE_FILE_ID}"
            response = session.get(url, stream=True)
            break
    if "html" in response.headers.get("Content-Type", ""):
        url = f"https://drive.usercontent.google.com/download?id={GDRIVE_FILE_ID}&export=download&confirm=t"
        response = session.get(url, stream=True)
    total = 0
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    size_mb = total / (1024 * 1024)
    print(f"   Model downloaded: {size_mb:.1f} MB")
    if size_mb < 1:
        os.remove(MODEL_PATH)
        raise RuntimeError("Download failed. Make sure file is shared as 'Anyone with the link' on Google Drive.")

IMG_SIZE = 224

# ── Class labels — exactly as Keras sorted them ──
CLASS_NAMES = [
    "Aloe_Vera",    # 0
    "Amla",         # 1
    "Ashwagandha",  # 2
    "Brahmi",       # 3
    "Curry_Leaf",   # 4
    "Giloy",        # 5
    "Moringa",      # 6
    "Neem",         # 7
    "Tulsi",        # 8
    "Turmeric"      # 9
]

# ══════════════════════════════════════════════
#  PLANT DATABASE — all 10 plants
#  Keys match CLASS_NAMES exactly
# ══════════════════════════════════════════════
PLANT_DB = {
    "Aloe_Vera": {
        "common_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "ayurvedic_name": "Ghritkumari / Kumari",
        "description": "Called the plant of immortality in ancient Egypt, Aloe Vera is widely used in Ayurveda for skin, digestion and immunity.",
        "parts_used": ["gel (inner leaf)", "latex (yellow layer)", "leaves"],
        "ayurvedic_benefits": [
            "Soothes and heals burns, wounds and skin irritation",
            "Deep moisturizes and nourishes skin",
            "Treats sunburn and reduces inflammation",
            "Improves digestion and treats constipation",
            "Boosts immunity with antioxidants",
            "Reduces blood sugar levels",
            "Promotes hair growth and treats dandruff",
            "Detoxifies the body and improves liver function"
        ],
        "how_to_use": {
            "gel_topical": "Extract fresh gel from leaf, apply directly on skin or hair",
            "aloe_juice": "Mix 2 tablespoons of fresh gel in water or juice and drink",
            "face_mask": "Apply fresh gel on face for 20 minutes then wash off",
            "hair_mask": "Apply gel on scalp and hair, leave for 30 minutes then wash",
            "burn_relief": "Apply fresh cold gel directly on burn or sunburn area"
        },
        "dosage": {
            "aloe_juice": "2–4 tablespoons of gel in water once a day",
            "topical": "Apply as needed on skin",
            "maximum_internal": "Do not consume more than 1–2 tablespoons of pure gel per day"
        },
        "toxicity": {
            "is_toxic": True,
            "toxic_parts": ["latex (yellow layer just under the skin)"],
            "warnings": [
                "⚠️ The yellow latex layer is TOXIC — always remove it completely",
                "Aloe latex can cause severe diarrhea and kidney damage",
                "Never consume aloe latex during pregnancy",
                "Do not consume internally if you have kidney disease"
            ]
        },
        "allergy_risks": [
            "Can cause skin rash in people allergic to plants in Liliaceae family",
            "Can cause low blood sugar — monitor if diabetic",
            "Patch test recommended before applying on face"
        ],
        "contraindications": [
            "Pregnancy — avoid internal consumption",
            "Kidney disease — avoid internal consumption",
            "Diabetes medications — monitor blood sugar closely",
            "Children under 12 — avoid internal consumption"
        ]
    },

    "Amla": {
        "common_name": "Amla",
        "scientific_name": "Phyllanthus emblica",
        "ayurvedic_name": "Amalaki",
        "description": "One of the most powerful Rasayana herbs in Ayurveda, Amla is rich in Vitamin C and used for longevity and immunity.",
        "parts_used": ["fruit", "seed", "leaves", "bark"],
        "ayurvedic_benefits": [
            "Rasayana (rejuvenative) herb used to enhance longevity and immunity",
            "Improves digestion and metabolism",
            "Supports liver function and detoxification",
            "Promotes hair growth and strengthens scalp health",
            "Helps manage blood sugar and cholesterol levels",
            "Strong antioxidant activity due to high Vitamin C and polyphenols"
        ],
        "how_to_use": {
            "fresh_fruit": "Eat fresh fruit directly or added to herbal preparations",
            "juice": "Dilute juice with water and consume in the morning",
            "powder": "Take Amla churna with warm water or honey",
            "formulations": "Used in classical formulations like Triphala and Chyawanprash",
            "oil": "Amla oil applied externally for hair nourishment"
        },
        "dosage": {
            "fruit_powder": "3–6 g per day",
            "fresh_juice": "10–20 ml per day",
            "decoction": "20–30 ml per day"
        },
        "toxicity": {
            "is_toxic": False,
            "toxic_parts": [],
            "warnings": [
                "Generally safe in recommended doses",
                "Large quantities may cause gastric irritation or acidity",
                "May enhance the effect of antidiabetic medication"
            ]
        },
        "allergy_risks": [
            "Rare cases of mild gastrointestinal discomfort",
            "Possible skin irritation in sensitive individuals"
        ],
        "contraindications": [
            "Diabetes medications — monitor blood sugar",
            "Avoid excess intake if prone to acidity"
        ]
    },

    "Ashwagandha": {
        "common_name": "Ashwagandha",
        "scientific_name": "Withania somnifera",
        "ayurvedic_name": "Ashwagandha / Vajikari",
        "description": "One of the most powerful herbs in Ayurveda, classified as a Rasayana (rejuvenator) and adaptogen.",
        "parts_used": ["root", "leaves", "seeds"],
        "ayurvedic_benefits": [
            "Reduces cortisol and stress levels (adaptogen)",
            "Improves strength, stamina and endurance",
            "Enhances memory and cognitive function",
            "Balances thyroid hormones",
            "Supports male fertility and testosterone",
            "Improves sleep quality",
            "Boosts immunity"
        ],
        "how_to_use": {
            "milk": "Mix 1 tsp powder in warm milk with honey at bedtime",
            "capsule": "Take capsule with water after meals",
            "powder": "Mix with ghee and honey for energy",
            "decoction": "Boil root in water and drink"
        },
        "dosage": {
            "powder": "3–6 grams daily with warm milk",
            "capsule": "300–500 mg twice daily",
            "extract": "250–600 mg standardized extract daily"
        },
        "toxicity": {
            "is_toxic": False,
            "toxic_parts": [],
            "warnings": [
                "May cause drowsiness — avoid driving initially",
                "Can cause stomach upset if taken on empty stomach",
                "Avoid during pregnancy",
                "High doses may cause diarrhea or vomiting"
            ]
        },
        "allergy_risks": [
            "May cause nasal congestion in some people",
            "Rare cases of skin rash reported"
        ],
        "contraindications": [
            "Pregnancy — avoid",
            "Thyroid medications — consult doctor",
            "Autoimmune disease — consult doctor",
            "Sedative medications — may increase effect"
        ]
    },

    "Brahmi": {
        "common_name": "Brahmi",
        "scientific_name": "Bacopa monnieri",
        "ayurvedic_name": "Brahmi / Saraswati",
        "description": "A powerful brain tonic used in Ayurveda for centuries to enhance memory, intelligence and calm the nervous system.",
        "parts_used": ["whole plant", "leaves", "stems"],
        "ayurvedic_benefits": [
            "Enhances memory and learning ability",
            "Reduces anxiety and mental stress",
            "Improves concentration and focus",
            "Neuroprotective — protects brain cells",
            "Helps in ADHD and cognitive disorders",
            "Promotes calmness and sleep quality",
            "Anti-epileptic properties"
        ],
        "how_to_use": {
            "juice": "Extract fresh juice and drink 10–20 ml daily",
            "powder": "Mix powder with warm milk or ghee",
            "oil": "Apply Brahmi oil on scalp for hair and brain health",
            "capsule": "Take capsule after meals with water",
            "tea": "Boil leaves in water and drink as herbal tea"
        },
        "dosage": {
            "fresh_juice": "10–20 ml per day",
            "powder": "3–6 g per day with milk",
            "capsule": "300–450 mg twice daily"
        },
        "toxicity": {
            "is_toxic": False,
            "toxic_parts": [],
            "warnings": [
                "May cause nausea, stomach cramps or diarrhea in some people",
                "Can cause fatigue or slow heart rate in high doses",
                "Avoid on empty stomach"
            ]
        },
        "allergy_risks": [
            "Rare digestive discomfort",
            "May cause increased bowel movements initially"
        ],
        "contraindications": [
            "Pregnancy — consult doctor",
            "Thyroid medications — may interact",
            "Sedatives — may increase effect",
            "Ulcer patients — take with food"
        ]
    },

    "Curry_Leaf": {
        "common_name": "Curry Leaf",
        "scientific_name": "Murraya koenigii",
        "ayurvedic_name": "Krishnanimba",
        "description": "Widely used in Indian cooking and Ayurveda, Curry Leaf is a potent medicinal herb for digestion, hair and blood sugar.",
        "parts_used": ["leaves", "roots", "bark"],
        "ayurvedic_benefits": [
            "Improves digestion and helps relieve diarrhea",
            "Possesses antioxidant and antimicrobial properties",
            "Supports blood sugar regulation",
            "Promotes healthy hair growth and scalp health",
            "Helps reduce cholesterol and support heart health"
        ],
        "how_to_use": {
            "raw": "Chew 8–10 fresh leaves every morning",
            "cooking": "Add fresh leaves to cooking for benefits",
            "powder": "Mix leaf powder with honey or water",
            "paste": "Apply leaf paste on scalp for hair growth"
        },
        "dosage": {
            "fresh_leaves": "8–12 leaves daily",
            "leaf_powder": "1–2 g daily",
            "leaf_juice": "10–15 ml daily"
        },
        "toxicity": {
            "is_toxic": False,
            "toxic_parts": [],
            "warnings": [
                "Considered safe in dietary amounts",
                "Very large consumption may cause mild gastrointestinal irritation"
            ]
        },
        "allergy_risks": [
            "Rare allergic reactions including itching or rash",
            "Sensitive individuals may experience digestive upset"
        ],
        "contraindications": [
            "Diabetes medications — monitor blood sugar",
            "Pregnancy — safe in food amounts, avoid medicinal doses"
        ]
    },

    "Giloy": {
        "common_name": "Giloy",
        "scientific_name": "Tinospora cordifolia",
        "ayurvedic_name": "Guduchi / Amrita",
        "description": "Known as Amrita (divine nectar) in Ayurveda, Giloy is one of the most potent immunomodulatory herbs.",
        "parts_used": ["stem", "roots", "leaves"],
        "ayurvedic_benefits": [
            "Powerful immunity booster and immunomodulator",
            "Reduces fever including chronic and dengue fever",
            "Detoxifies blood and liver",
            "Anti-inflammatory — helps in arthritis",
            "Improves digestion and treats constipation",
            "Manages diabetes by regulating blood sugar",
            "Reduces stress and anxiety",
            "Anti-cancer properties under research"
        ],
        "how_to_use": {
            "juice": "Extract stem juice and drink 20–30 ml daily",
            "kadha": "Boil stem pieces in water and drink as decoction",
            "powder": "Mix powder with honey or warm water",
            "capsule": "Take capsule after meals"
        },
        "dosage": {
            "stem_juice": "20–30 ml per day",
            "powder": "2–4 g per day",
            "capsule": "500 mg twice daily"
        },
        "toxicity": {
            "is_toxic": False,
            "toxic_parts": [],
            "warnings": [
                "May lower blood sugar significantly — monitor if diabetic",
                "Avoid during pregnancy and breastfeeding",
                "May cause constipation in some people",
                "Avoid if you have autoimmune disease"
            ]
        },
        "allergy_risks": [
            "Rare allergic skin reactions",
            "May cause mild digestive discomfort initially"
        ],
        "contraindications": [
            "Autoimmune conditions — may overstimulate immune system",
            "Pregnancy — avoid",
            "Diabetes medications — monitor blood sugar closely",
            "Surgery — stop 2 weeks before scheduled surgery"
        ]
    },

    "Moringa": {
        "common_name": "Moringa",
        "scientific_name": "Moringa oleifera",
        "ayurvedic_name": "Shigru",
        "description": "Called the miracle tree, Moringa is one of the most nutrient-dense plants used in Ayurveda for energy and healing.",
        "parts_used": ["leaves", "pods", "seeds", "flowers", "root bark"],
        "ayurvedic_benefits": [
            "Anti-inflammatory and analgesic for joint pain",
            "Rich nutritional profile — Vitamins A, C and minerals",
            "Helps regulate blood glucose levels",
            "Supports cardiovascular health",
            "Boosts immunity and energy levels",
            "Promotes digestive health"
        ],
        "how_to_use": {
            "leaves": "Cook leaves as vegetables or add to soups",
            "powder": "Add leaf powder to smoothies or water",
            "pods": "Use pods in traditional Indian dishes",
            "tea": "Boil leaves in water as herbal tea",
            "oil": "Seed oil used in skincare preparations"
        },
        "dosage": {
            "leaf_powder": "2–5 g daily",
            "leaf_decoction": "30–50 ml daily",
            "fresh_leaves": "1 cup cooked leaves per day"
        },
        "toxicity": {
            "is_toxic": True,
            "toxic_parts": ["root bark", "roots in large quantities"],
            "warnings": [
                "Root and root bark contain alkaloids — toxic in large quantities",
                "Excess consumption may cause digestive disturbances",
                "May reduce blood pressure significantly with antihypertensive drugs"
            ]
        },
        "allergy_risks": [
            "Occasional digestive discomfort",
            "Rare skin reactions in sensitive individuals"
        ],
        "contraindications": [
            "Pregnancy — avoid root and bark",
            "Blood pressure medications — monitor closely",
            "Thyroid medications — may interact"
        ]
    },

    "Neem": {
        "common_name": "Neem",
        "scientific_name": "Azadirachta indica",
        "ayurvedic_name": "Nimba",
        "description": "Known as the village pharmacy of India, Neem is one of the most versatile medicinal plants used in Ayurveda for over 4000 years.",
        "parts_used": ["leaves", "bark", "seeds", "oil", "flowers", "roots"],
        "ayurvedic_benefits": [
            "Powerful antibacterial and antifungal properties",
            "Purifies blood and removes toxins",
            "Treats skin diseases — eczema, psoriasis, acne",
            "Controls blood sugar levels in diabetics",
            "Strengthens teeth and gums",
            "Boosts liver health and detoxification",
            "Natural pesticide — repels insects and parasites",
            "Reduces inflammation and joint pain"
        ],
        "how_to_use": {
            "leaves_raw": "Chew 4–5 fresh neem leaves every morning",
            "neem_water": "Boil leaves in water, cool and use for skin wash",
            "neem_paste": "Grind leaves with water, apply on skin for infections",
            "neem_oil": "Apply diluted neem oil on scalp for dandruff",
            "bark_decoction": "Boil bark in water and drink for fever"
        },
        "dosage": {
            "leaves_raw": "4–5 leaves per day maximum",
            "neem_juice": "10–15 ml once a day",
            "neem_powder": "1–3 grams per day"
        },
        "toxicity": {
            "is_toxic": True,
            "toxic_parts": ["seeds", "seed oil in large quantities"],
            "warnings": [
                "⚠️ TOXIC TO CHILDREN — Neem oil is highly toxic to infants",
                "⚠️ Never give neem oil internally to children under 12",
                "Excessive consumption can cause vomiting, seizures and liver damage",
                "Do not consume during pregnancy — may cause miscarriage"
            ]
        },
        "allergy_risks": [
            "Can cause skin rash in sensitive individuals",
            "May trigger reaction in people with tree nut allergies",
            "Neem oil can cause eye irritation"
        ],
        "contraindications": [
            "Pregnancy — strictly avoid",
            "Children under 12 — avoid neem oil internally",
            "Autoimmune conditions — may overstimulate immune system",
            "Organ transplant patients — avoid"
        ]
    },

    "Tulsi": {
        "common_name": "Tulsi",
        "scientific_name": "Ocimum sanctum",
        "ayurvedic_name": "Tulsi / Vrinda",
        "description": "One of the most sacred and medicinally important plants in Ayurveda, known as the Queen of Herbs.",
        "parts_used": ["leaves", "seeds", "roots", "stem"],
        "ayurvedic_benefits": [
            "Boosts immunity and fights infections",
            "Relieves cough, cold and respiratory disorders",
            "Reduces fever and inflammation",
            "Improves digestion and gut health",
            "Reduces stress and anxiety (adaptogen)",
            "Purifies blood and detoxifies body",
            "Improves skin health and treats acne",
            "Helps manage diabetes by regulating blood sugar"
        ],
        "how_to_use": {
            "tea": "Boil 10–12 fresh Tulsi leaves in water for 5 minutes, add honey and drink",
            "raw": "Chew 4–5 fresh leaves every morning on empty stomach",
            "paste": "Grind leaves into paste and apply on skin for acne",
            "steam": "Add leaves to hot water and inhale steam for cold"
        },
        "dosage": {
            "leaves_raw": "4–5 leaves per day",
            "tulsi_tea": "2–3 cups per day",
            "tulsi_juice": "10–20 ml twice a day"
        },
        "toxicity": {
            "is_toxic": False,
            "toxic_parts": [],
            "warnings": [
                "Avoid large quantities during pregnancy",
                "May thin blood — avoid before surgery",
                "Long term excessive use may affect fertility",
                "Avoid giving to infants below 1 year"
            ]
        },
        "allergy_risks": [
            "People allergic to mint family plants may react",
            "May cause mild skin irritation in sensitive individuals"
        ],
        "contraindications": [
            "Pregnancy — avoid large doses",
            "Blood thinning medications — consult doctor",
            "Hypoglycemia — monitor blood sugar levels"
        ]
    },

    "Turmeric": {
        "common_name": "Turmeric",
        "scientific_name": "Curcuma longa",
        "ayurvedic_name": "Haridra",
        "description": "Golden spice and potent medicinal herb used in Ayurveda for 4000+ years for healing and immunity.",
        "parts_used": ["rhizome"],
        "ayurvedic_benefits": [
            "Potent anti-inflammatory and antioxidant herb",
            "Supports immune function",
            "Used for wound healing and skin disorders",
            "Improves digestion and liver function",
            "May help manage arthritis and joint inflammation",
            "Anti-cancer properties under research"
        ],
        "how_to_use": {
            "cooking": "Add turmeric powder in cooking daily",
            "golden_milk": "Mix 1 tsp in warm milk with black pepper at night",
            "paste": "Apply turmeric paste externally for wounds or skin",
            "decoction": "Boil in water and drink as herbal tea"
        },
        "dosage": {
            "powder": "1–3 g per day",
            "curcumin_extract": "500–1000 mg per day",
            "decoction": "20–30 ml per day"
        },
        "toxicity": {
            "is_toxic": False,
            "toxic_parts": [],
            "warnings": [
                "High doses may cause gastrointestinal irritation",
                "Individuals with gallbladder disease should avoid high intake",
                "May interact with blood thinning medications",
                "Always take with black pepper to increase absorption"
            ]
        },
        "allergy_risks": [
            "Possible contact dermatitis when applied to skin",
            "Rare digestive upset or nausea in high doses"
        ],
        "contraindications": [
            "Gallbladder disease — avoid high doses",
            "Blood thinners — consult doctor",
            "Pregnancy — safe in food amounts, avoid supplements",
            "Iron deficiency — may reduce iron absorption"
        ]
    }
}


# ══════════════════════════════════════════════
#  LOAD MODEL AT STARTUP
# ══════════════════════════════════════════════
print("\n🌿 AyurScan starting...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n❌ Model not found: '{MODEL_PATH}'\n"
        f"Place your .h5 file in the same folder as app.py\n"
        f"Available .h5 files: ayurscan_81percent_BEST.h5"
    )

download_model_from_drive()
print(f"   Loading model: {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"   ✅ Model ready | classes: {len(CLASS_NAMES)}")
print(f"   ✅ Plants DB  : {list(PLANT_DB.keys())}\n")


# ══════════════════════════════════════════════
#  IMAGE PREPROCESSING
#  Matches training exactly:
#  preprocessing_function = preprocess_input (EfficientNet)
#  target_size = (224, 224)
# ══════════════════════════════════════════════
def prepare_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)               # EfficientNet preprocessing
    return np.expand_dims(arr, axis=0)        # (1, 224, 224, 3)


# ══════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "AyurScan API",
        "status":  "online",
        "plants":  CLASS_NAMES
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "running",
        "model":       MODEL_PATH,
        "num_classes": len(CLASS_NAMES),
        "classes":     CLASS_NAMES
    })


@app.route("/api/scan", methods=["POST"])
def scan():
    """
    POST /api/scan
    ──────────────
    Form-data key : 'image'  (jpg / png / webp)

    Response:
    {
        "success"    : true,
        "label"      : "Tulsi",
        "confidence" : 91.23,
        "plant"      : { ...full plant info from database... },
        "top3"       : [
            {"label": "Tulsi",    "confidence": 91.23},
            {"label": "Neem",     "confidence":  5.12},
            {"label": "Moringa",  "confidence":  3.65}
        ],
        "image_data" : "data:image/jpeg;base64,..."
    }
    """

    if "image" not in request.files:
        return jsonify({"error": "No image sent. Use form-data with key 'image'"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    if ext not in {"jpg", "jpeg", "png", "webp", "bmp"}:
        return jsonify({"error": f"Unsupported file type .{ext}"}), 400

    try:
        raw   = file.read()
        inp   = prepare_image(raw)
        preds = model.predict(inp, verbose=0)[0]   # shape: (10,)

        top_idx    = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        label      = CLASS_NAMES[top_idx]

        # Top 3 predictions
        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [
            {
                "label":      CLASS_NAMES[i],
                "confidence": round(float(preds[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        # Get full plant info from database
        plant_info = PLANT_DB.get(label, {})

        # Encode image as base64 for frontend display
        mime    = f"image/{'jpeg' if ext == 'jpg' else ext}"
        img_b64 = base64.b64encode(raw).decode("utf-8")

        return jsonify({
            "success":    True,
            "label":      label,
            "confidence": round(confidence * 100, 2),
            "plant":      plant_info,
            "top3":       top3,
            "image_data": f"data:{mime};base64,{img_b64}"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/plants", methods=["GET"])
def all_plants():
    """List all plants with basic info"""
    return jsonify({
        "total":  len(PLANT_DB),
        "plants": [
            {
                "key":             k,
                "common_name":     v["common_name"],
                "scientific_name": v["scientific_name"],
                "ayurvedic_name":  v["ayurvedic_name"]
            }
            for k, v in PLANT_DB.items()
        ]
    })


@app.route("/api/plants/<n>", methods=["GET"])
def get_plant(n):
    """Get full info for one plant — e.g. /api/plants/Tulsi"""
    if n not in PLANT_DB:
        return jsonify({"error": f"Plant '{n}' not found. Available: {CLASS_NAMES}"}), 404
    return jsonify(PLANT_DB[n])


# ══════════════════════════════════════════════
#  RUN  (Render uses Procfile / gunicorn)
# ══════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

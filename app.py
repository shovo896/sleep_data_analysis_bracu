from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


DATA_FILE = "Sleep_health_and_lifestyle_dataset (1).csv"
TARGET_COLUMN = "Sleep Disorder"


def load_and_clean_data() -> pd.DataFrame:
    """Load and apply the same feature preparation used in voting_cls.ipynb."""
    data_path = Path(__file__).resolve().parent / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Upload {DATA_FILE} to the Space repository."
        )

    df = pd.read_csv(data_path).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna("None")

    df = df.drop(columns=["Person ID"], errors="ignore")
    if "BMI Category" in df.columns:
        df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

    if "Blood Pressure" in df.columns:
        blood_pressure = df["Blood Pressure"].astype(str).str.split("/", expand=True)
        df["Systolic_BP"] = pd.to_numeric(blood_pressure[0], errors="coerce")
        df["Diastolic_BP"] = pd.to_numeric(blood_pressure[1], errors="coerce")
        df = df.drop(columns=["Blood Pressure"])

    return df.dropna().reset_index(drop=True)


def build_voting_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Build a lightweight tree-ensemble soft-voting pipeline."""
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
    extra_trees = ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
    )
    voting_classifier = VotingClassifier(
        estimators=[("rf", random_forest), ("extra", extra_trees)],
        voting="soft",
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", voting_classifier)]
    )
    model.fit(X, y)
    return model


@lru_cache(maxsize=1)
def get_artifacts() -> dict[str, Any]:
    df = load_and_clean_data()
    X = df.drop(columns=[TARGET_COLUMN])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])
    model = build_voting_model(X, y)

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_choices = {
        column: sorted(X[column].astype(str).unique().tolist())
        for column in categorical_columns
    }
    numerical_defaults = {
        column: {
            "minimum": float(X[column].min()),
            "maximum": float(X[column].max()),
            "value": float(X[column].median()),
            "step": 1 if pd.api.types.is_integer_dtype(X[column]) else 0.1,
        }
        for column in numerical_columns
    }

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "categorical_choices": categorical_choices,
        "numerical_defaults": numerical_defaults,
        "examples": X.sample(n=min(3, len(X)), random_state=42).values.tolist(),
    }


def predict_sleep_disorder(*values: Any) -> tuple[str, dict[str, float]]:
    artifacts = get_artifacts()
    input_df = pd.DataFrame([values], columns=artifacts["feature_columns"])

    model: Pipeline = artifacts["model"]
    label_encoder: LabelEncoder = artifacts["label_encoder"]
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    label = label_encoder.inverse_transform([prediction])[0]
    class_names = label_encoder.inverse_transform(range(len(probabilities)))
    scores = {
        str(class_name): float(probability)
        for class_name, probability in zip(class_names, probabilities)
    }
    return str(label), scores


app = FastAPI(title="Sleep Disorder Prediction")


def render_page() -> str:
    artifacts = get_artifacts()
    fields: list[str] = []
    for column in artifacts["feature_columns"]:
        if column in artifacts["categorical_choices"]:
            options = "".join(
                f'<option value="{value}">{value}</option>'
                for value in artifacts["categorical_choices"][column]
            )
            control = f'<select name="{column}">{options}</select>'
        else:
            settings = artifacts["numerical_defaults"][column]
            control = (
                f'<input type="number" name="{column}" value="{settings["value"]}" '
                f'min="{settings["minimum"]}" max="{settings["maximum"]}" '
                f'step="{settings["step"]}" required>'
            )
        fields.append(f'<label><span>{column}</span>{control}</label>')

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sleep Disorder Prediction</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;font-family:Inter,system-ui,sans-serif;background:#07111f;color:#eaf2ff}}
.wrap{{max-width:1050px;margin:auto;padding:48px 20px}} h1{{font-size:clamp(2rem,5vw,3.6rem);margin:0 0 10px}}
.sub{{color:#a9b8cf;max-width:720px;line-height:1.6}} .card{{margin-top:28px;background:#101e31;border:1px solid #263a54;border-radius:22px;padding:24px;box-shadow:0 24px 70px #0006}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}} label span{{display:block;font-size:.85rem;color:#b8c5d8;margin:0 0 7px}}
input,select{{width:100%;padding:12px;border-radius:10px;border:1px solid #334963;background:#091525;color:#f5f8ff;font:inherit}}
button{{margin-top:22px;width:100%;padding:14px;border:0;border-radius:12px;background:linear-gradient(135deg,#54d6ff,#7772ff);color:#06111d;font-weight:800;font-size:1rem;cursor:pointer}}
#result{{display:none;margin-top:20px;padding:18px;border-radius:14px;background:#0a1728}} .label{{font-size:1.5rem;font-weight:800;color:#62dcff}}
.bar{{height:9px;background:#203149;border-radius:9px;overflow:hidden;margin:5px 0 12px}} .fill{{height:100%;background:#6f8cff}} .note{{font-size:.8rem;color:#8292a9;margin-top:18px}}
</style></head><body><main class="wrap"><h1>Sleep Disorder Prediction</h1>
<p class="sub">Enter lifestyle and biometric information to obtain a prediction from a tree-ensemble soft-voting model.</p>
<section class="card"><form id="form"><div class="grid">{"".join(fields)}</div><button type="submit">Predict Sleep Disorder</button></form>
<div id="result"><div>Prediction</div><div class="label" id="prediction"></div><div id="scores"></div></div>
<p class="note">For educational use only; this is not medical advice or a clinical diagnosis.</p></section></main>
<script>
const form=document.querySelector('#form'), result=document.querySelector('#result');
form.addEventListener('submit',async(e)=>{{e.preventDefault();const button=form.querySelector('button');button.disabled=true;button.textContent='Predicting…';
 const payload=Object.fromEntries(new FormData(form));
 try{{const response=await fetch('/api/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Prediction failed');
 document.querySelector('#prediction').textContent=data.prediction;document.querySelector('#scores').innerHTML=Object.entries(data.probabilities).sort((a,b)=>b[1]-a[1]).map(([name,value])=>`<div>${{name}} — ${{(value*100).toFixed(1)}}%</div><div class="bar"><div class="fill" style="width:${{value*100}}%"></div></div>`).join('');result.style.display='block';
 }}catch(error){{alert(error.message)}}finally{{button.disabled=false;button.textContent='Predict Sleep Disorder'}}
}});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return render_page()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = get_artifacts()
    try:
        values = []
        for column in artifacts["feature_columns"]:
            value = payload[column]
            values.append(
                str(value)
                if column in artifacts["categorical_choices"]
                else float(value)
            )
        label, scores = predict_sleep_disorder(*values)
        return {"prediction": label, "probabilities": scores}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid input: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

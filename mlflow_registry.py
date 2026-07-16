"""
mlflow_registry.py — Enregistrement du pipeline OCR+LLM dans le MLflow Model Registry.

Le "modele" ici n'est pas un modele entraine mais la configuration du pipeline
(moteur OCR + LLM + prompt). On l'enregistre quand meme comme pyfunc pour
beneficier du Model Registry : versions, stages (Staging/Production), et
metriques/params attaches automatiquement via le run_id.

Usage typique (appele depuis mlops_docling.py a la fin d'un run) :
    from mlflow_registry import register_pipeline_model
    register_pipeline_model(
        engine="docling", llm_model="llama3.1", prompt_version="v3_snake_case_keys_only",
        accuracy_pct=summary["global_accuracy_pct"],
        registered_model_name="pfa-ocr-pipeline",
    )
"""
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


class OCRLLMPipelineWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper pyfunc: encapsule la config du pipeline (moteur OCR + LLM + prompt).
    predict() appelle l'API FastAPI (upload -> ocr -> extract) pour un lot de PDFs.
    """

    def __init__(self, engine: str, llm_model: str, prompt_version: str, api_url: str = "http://localhost:8000"):
        self.engine = engine
        self.llm_model = llm_model
        self.prompt_version = prompt_version
        self.api_url = api_url

    def predict(self, context, model_input):
        import requests
        results = []
        for pdf_path in model_input:
            with open(pdf_path, "rb") as f:
                r = requests.post(f"{self.api_url}/upload", files={"file": (pdf_path, f, "application/pdf")})
            r.raise_for_status()
            doc_id = r.json()["document_id"]
            requests.post(f"{self.api_url}/ocr/{doc_id}").raise_for_status()
            r = requests.post(f"{self.api_url}/extract/{doc_id}")
            r.raise_for_status()
            results.append(r.json().get("extracted_data", {}))
        return results


def register_pipeline_model(
    engine: str,
    llm_model: str,
    prompt_version: str,
    accuracy_pct: float,
    registered_model_name: str = "pfa-ocr-pipeline",
    stage: str | None = None,
):
    """
    A appeler DANS un `with mlflow.start_run(...)` deja ouvert (reutilise le run actif),
    pour que le modele enregistre soit lie au run et a ses metriques/params.

    - log_model() : sauvegarde le modele comme artefact du run.
    - registered_model_name= : cree automatiquement une nouvelle VERSION dans le
      Model Registry (cree le "registered model" s'il n'existe pas encore).
    - stage : optionnel, transitionne la nouvelle version vers "Staging" ou
      "Production" (ex: seulement si accuracy_pct depasse un seuil).
    """
    wrapper = OCRLLMPipelineWrapper(engine=engine, llm_model=llm_model, prompt_version=prompt_version)

    model_info = mlflow.pyfunc.log_model(
        artifact_path="ocr_llm_pipeline",
        python_model=wrapper,
        registered_model_name=registered_model_name,
        pip_requirements=["requests"],
    )

    client = MlflowClient()
    version = model_info.registered_model_version

    # Tags utiles pour retrouver/filtrer les versions dans l'UI du registre
    client.set_model_version_tag(registered_model_name, version, "engine", engine)
    client.set_model_version_tag(registered_model_name, version, "llm_model", llm_model)
    client.set_model_version_tag(registered_model_name, version, "prompt_version", prompt_version)
    client.set_model_version_tag(registered_model_name, version, "accuracy_pct", str(accuracy_pct))

    if stage:
        client.transition_model_version_stage(
            name=registered_model_name, version=version, stage=stage
        )

    print(f"Modele enregistre : {registered_model_name} v{version} (accuracy={accuracy_pct}%, stage={stage})")
    return version
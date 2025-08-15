import os
import sys
import traceback

import joblib
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi, upload_file, whoami
from pydantic import BaseModel

from data_collection.combine_features import combine_features
from data_collection.generate_deposit_reuse_pairs import generate_deposit_reuse_pairs
from data_collection.node_embedding_exporter import node_embedding_exporter
from data_collection.time_amount_exporter import time_amount_exporter
from model.initiate_model import LightGBMTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app = FastAPI()


# ---- Request schema ----
class PipelineRequest(BaseModel):
    chain: str = "ethereum"
    radius: int = 2
    batch_size: int = 3600
    max_workers: int = 4


@app.post("/run-pipeline")
def run_pipeline(req: PipelineRequest):
    try:
        working_dir = "output"
        os.makedirs(working_dir, exist_ok=True)

        # Step 1: Export time-amount features
        time_amount_exporter(
            chain=req.chain,
            radius=req.radius,
            saving_path=working_dir,
            batch_size=req.batch_size,
            max_workers=req.max_workers,
        )
        from_path = os.path.join(working_dir, "from_df.csv")
        to_path = os.path.join(working_dir, "to_df.csv")

        # Step 2: Export node embeddings
        node_embedding_exporter(
            chain=req.chain,
            radius=req.radius,
            saving_path=working_dir,
        )
        embedding_path = os.path.join(working_dir, "embedding_df.csv")

        # Step 3: Generate deposit reuse pairs
        pair_path = os.path.join(working_dir, "deposit_pairs.csv")
        generate_deposit_reuse_pairs(
            chain_name="ethereum",
            file_path=pair_path,
            max_workers=req.max_workers,
            batch_size=2000,
        )

        # Step 4: Combine all features into training dataset
        combine_features(
            from_path=from_path,
            to_path=to_path,
            embedding_path=embedding_path,
            pair_path=pair_path,
            saving_path=working_dir,
            max_workers=req.max_workers,
            batch_size=req.batch_size,
            compute_embedding_similarity=False,  # Set to False to skip embedding similarity computation
        )

        # Step 5: Train and evaluate model
        train_file = os.path.join(working_dir, "train_data.csv")
        test_file = os.path.join(working_dir, "test_data.csv")
        drop_columns = ["Unnamed: 0", "Diff2_Vec_Simi"]

        trainer = LightGBMTrainer(drop_cols=drop_columns)
        model, results = trainer.train_and_evaluate(train_file, test_file)

        # Step 6: Save and upload model to Hugging Face
        # Step 6: Save and upload model to Hugging Face (robust)

        model_path = os.path.join(working_dir, "lightgbm_model.pkl")
        joblib.dump(model, model_path)

        # 1) Lấy token (đồng nhất 1 tên biến env)
        api_key = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("HF_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Missing HF token. Set HF_TOKEN (or HUGGINGFACE_API_KEY/HF_API_KEY)."
            )

        api = HfApi(token=api_key)

        # 2) Xác định namespace
        who = whoami(token=api_key)
        user_name = who.get("name")  # username của token
        # Nếu muốn dùng org, set org_name = "your-org"; nếu không, để None
        org_name = None  # ví dụ: "my-org" nếu muốn tạo repo trong org

        repo_basename = f"{req.chain}-reuse-model"
        if org_name:
            repo_id = f"{org_name}/{repo_basename}"
        else:
            repo_id = f"{user_name}/{repo_basename}"

        # 3) Tạo repo (nếu cần) đúng namespace + loại "model"
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=False,  # tuỳ bạn
                exist_ok=True,
                organization=org_name,  # None nếu đẩy lên user; để "my-org" nếu lên org
            )
        except Exception as e:
            # Nếu đã tồn tại hoặc quyền không cho, vẫn cứ thử upload — nhưng log ra cho rõ
            print(f"[WARN] create_repo: {e}")

        # 4) (Tuỳ chọn) xác nhận repo có tồn tại sau khi create
        try:
            api.repo_info(repo_id=repo_id, repo_type="model")
        except Exception as e:
            raise RuntimeError(
                f"Repo not accessible after create: {repo_id}. Error: {e}"
            )

        # 5) Upload file dùng cùng token & repo_id namespaced
        upload_file(
            path_or_fileobj=model_path,
            path_in_repo="lightgbm_model.pkl",
            repo_id=repo_id,
            repo_type="model",
            token=api_key,
        )

        return {
            "message": "Pipeline and model training completed successfully.",
            "evaluation": {
                "f1": results["f1"],
                "accuracy": results["accuracy"],
                "report": results["classification_report"],
            },
            "huggingface_model_url": f"https://huggingface.co/{repo_id}",
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# # src/ingestion/kaggle_ingest.py
# import pandas as pd
# import os
# from datetime import datetime

# from validate import is_valid_image
# from normalize import normalize_image
# import config
# from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[2]


# def run():
#     df = pd.read_csv(config.RAW_METADATA_PATH,sep=',',on_bad_lines="skip")

#     records = []
#     skipped = 0

#     for _, row in df.iterrows():
#         product_id = str(row["id"])
#         image_name = f"{product_id}.jpg"
#         raw_image_path = os.path.join(config.RAW_IMAGE_DIR, image_name)

#         if not os.path.exists(raw_image_path):
#             skipped += 1
#             continue

#         if not is_valid_image(raw_image_path):
#             skipped += 1
#             continue

#         output_image_path = os.path.join(
#             config.PROCESSED_IMAGE_DIR,
#             image_name
#         )

#         normalize_image(
#             raw_image_path,
#             output_image_path,
#             config.IMAGE_SIZE
#         )

#         records.append({
#             "product_id": product_id,
#             "source": config.SOURCE_NAME,
#             "image_path": output_image_path,
#             "category": row["articleType"],
#             "brand": row.get("brand"),
#             "price": None,
#             "currency": None,
#             "ingestion_timestamp": datetime.utcnow().isoformat()
#         })

#     os.makedirs(os.path.dirname(config.PROCESSED_METADATA_PATH), exist_ok=True)
#     pd.DataFrame(records).to_csv(
#         config.PROCESSED_METADATA_PATH,
#         index=False
#     )

#     print(f"Valid products: {len(records)}")
#     print(f"Skipped records: {skipped}")

# if __name__ == "__main__":
#     run()
# src/ingestion/kaggle_ingest.py
import pandas as pd
from datetime import datetime,UTC
from pathlib import Path

from validate import is_valid_image
from normalize import normalize_image
import config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run():
    # --- Read CSV SAFELY ---
    df = pd.read_csv(
        config.RAW_METADATA_PATH,
        sep=",",
        engine="python",
        on_bad_lines="skip"
    )

    records = []
    skipped = 0

    for _, row in df.iterrows():
        product_id = str(row["id"])
        image_name = f"{product_id}.jpg"

        raw_image_path = PROJECT_ROOT / config.RAW_IMAGE_DIR / image_name

        if not raw_image_path.exists():
            skipped += 1
            continue

        if not is_valid_image(raw_image_path):
            skipped += 1
            continue

        output_image_path = PROJECT_ROOT / config.PROCESSED_IMAGE_DIR / image_name
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        normalize_image(
            raw_image_path,
            output_image_path,
            config.IMAGE_SIZE
        )

        # records.append({
        #     "product_id": product_id,
        #     "source": config.SOURCE_NAME,
        #     "image_path": str(output_image_path.relative_to(PROJECT_ROOT)),
        #     "category": row["articleType"],
        #     "brand": row.get("brand"),
        #     "price": None,
        #     "currency": None,
        #     "ingestion_timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat()
        # })
        records.append({
            "product_id": product_id,
            "source": config.SOURCE_NAME,
            "image_path": f"data/processed/images/{image_name}",
            "category": row["articleType"],
            "brand": row.get("brand"),
            "productDisplayName": row["productDisplayName"],  # ← ADD THIS
            "price": None,
            "currency": None,
            "ingestion_timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat()
        })


    print(f"Valid products before safety check: {len(records)}")
    print(f"Skipped records: {skipped}")

    # --- HARD SAFETY CHECK ---
    if len(records) < 10000:
        raise RuntimeError(
            f"FATAL: Only {len(records)} records ingested. "
            "Refusing to overwrite metadata.csv"
        )

    metadata_path = PROJECT_ROOT / config.PROCESSED_METADATA_PATH
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(metadata_path, index=False)

    print(f"✅ Ingestion complete: {len(records)} records")


if __name__ == "__main__":
    run()

import os
import csv
import Levenshtein
import lmstudio as lms
from tqdm import tqdm

# KONFIGURASI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_ROOT = os.path.join(BASE_DIR, "Indonesian License Plate Recognition")

IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")

RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RESULT_DIR, "hasil_ocr.csv")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

VLM_MODEL_NAME = "qwen/qwen2.5-vl-7b"

PROMPT_TEXT = (
    "Read the Indonesian license plate in this image. "
    "Return ONLY the plate number without explanation."
)

# CER

def levenshtein_ops(gt: str, pred: str):
    ops = Levenshtein.editops(gt, pred)

    S = sum(1 for op, _, _ in ops if op == "replace")
    D = sum(1 for op, _, _ in ops if op == "delete")
    I = sum(1 for op, _, _ in ops if op == "insert")

    return S, D, I


def calculate_cer(gt: str, pred: str):

    if len(gt) == 0:
        return "0.00%"

    S, D, I = levenshtein_ops(gt, pred)

    cer = (S + D + I) / len(gt)

    return f"{cer*100:.2f}%"


# LOAD GROUND TRUTH

def load_ground_truth(images_dir, labels_dir):

    gt = {}

    if not os.path.isdir(images_dir):
        print(f"\nFolder gambar tidak ditemukan:\n{images_dir}")
        return gt

    if not os.path.isdir(labels_dir):
        print(f"\nFolder label tidak ditemukan:\n{labels_dir}")
        return gt

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    print(f"Jumlah gambar ditemukan : {len(image_files)}")

    for image_name in image_files:

        txt_name = os.path.splitext(image_name)[0] + ".txt"

        txt_path = os.path.join(labels_dir, txt_name)

        if not os.path.exists(txt_path):
            print(f"Label tidak ada : {txt_name}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if text == "":
            print(f"Label kosong : {txt_name}")
            continue

        gt[image_name] = text.replace(" ", "").upper()

    return gt


# OCR

def ocr_image(model, image_path):

    try:

        image = lms.prepare_image(image_path)

        chat = lms.Chat()

        chat.add_user_message(
            PROMPT_TEXT,
            images=[image]
        )

        response = model.respond(chat)

        try:
            result = response.content
        except AttributeError:
            result = str(response)

        result = (
            result
            .replace(" ", "")
            .replace("\n", "")
            .upper()
            .strip()
        )

        return result

    except Exception as e:

        print(f"\nERROR : {os.path.basename(image_path)}")
        print(e)

        return "ERROR"


# MAIN

def main():

    print("=" * 60)
    print("OCR INDONESIAN LICENSE PLATE")
    print("=" * 60)

    print("\nBASE DIR")
    print(BASE_DIR)

    print("\nIMAGE DIR")
    print(IMAGES_DIR)

    print("\nLABEL DIR")
    print(LABELS_DIR)

    print("\nOUTPUT")
    print(OUTPUT_CSV)

    print("\nChecking dataset...")

    ground_truth = load_ground_truth(IMAGES_DIR, LABELS_DIR)

    print(f"Ground Truth Loaded : {len(ground_truth)}")

    if len(ground_truth) == 0:
        print("\nTidak ada data.")
        return

    print("\nLoading LM Studio model...")

    try:

        model = lms.llm(VLM_MODEL_NAME)

        print("Model berhasil dimuat.")

    except Exception as e:

        print("\nTidak dapat memuat model LM Studio.")
        print(e)
        return

    image_files = list(ground_truth.keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "image",
            "ground_truth",
            "prediction",
            "CER_score"
        ])

        for image_name in tqdm(image_files):

            image_path = os.path.join(IMAGES_DIR, image_name)

            gt = ground_truth[image_name]

            pred = ocr_image(model, image_path)

            cer = calculate_cer(gt, pred)

            writer.writerow([
                image_name,
                gt,
                pred,
                cer
            ])

            print("-" * 60)
            print("Image :", image_name)
            print("GT    :", gt)
            print("Pred  :", pred)
            print("CER_score   :", cer)

    print("\n====================================================")
    print("SELESAI")
    print("====================================================")
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
Nama : Mirna Rahmania Melba  
NIM  : 4222301010
Kelas: RE-6A Pagi

# OCR Plat Nomor Indonesia Menggunakan Vision Language Model (VLM)
Program ini merupakan sistem **Optical Character Recognition (OCR)** untuk mengenali plat nomor kendaraan Indonesia menggunakan **Vision Language Model (VLM)** yang dijalankan melalui **LM Studio** dan diintegrasikan dengan **Python**.

Berbeda dengan OCR tradisional yang memerlukan proses preprocessing, segmentasi karakter, dan klasifikasi karakter, pendekatan ini memanfaatkan kemampuan Vision Language Model untuk memahami gambar secara langsung melalui prompt sehingga menghasilkan teks plat nomor.

Model yang digunakan adalah (qwen/qwen2.5-vl-7b)

Program akan secara otomatis:
- Membaca seluruh gambar pada dataset
- Membaca ground truth dari file label (.txt)
- Mengirim gambar ke LM Studio
- Melakukan inferensi OCR menggunakan Vision Language Model
- Menghitung Character Error Rate (CER)
- Menyimpan seluruh hasil ke file CSV

# Overview
Project ini bertujuan melakukan OCR pada plat nomor kendaraan Indonesia menggunakan Vision Language Model (VLM). Seluruh proses inferensi dijalankan secara lokal menggunakan LM Studio, sedangkan Python bertugas membaca dataset, mengirim gambar ke model, menerima hasil prediksi, menghitung CER, dan menyimpan hasil evaluasi.

# Features
- Vision Language Model (Qwen2.5-VL-7B)
- OCR Plat Nomor Indonesia
- Local Inference menggunakan LM Studio
- Automatic Dataset Processing
- Character Error Rate (CER) Evaluation
- CSV Result Export

# Struktur Folder
Project
│
├── AAS_CV.py
│
├── result
│     └── hasil_ocr.csv
│
├── Indonesian License Plate Recognition
│     ├── images/
│     └── labels/
│
└── README.md

# Workflow
Dataset Images
        │
        ▼
Load Ground Truth
        │
        ▼
Python Program
        │
        ▼
LM Studio
(Qwen2.5-VL-7B)
        │
        ▼
Vision Language Model
        │
        ▼
OCR Prediction
        │
        ▼
Character Error Rate
        │
        ▼
hasil_ocr.csv

# Requirements
- Python 3.10+
- LM Studio
- Qwen2.5-VL-7B
- CUDA GPU (Recommended)

# Install Dependencies
Install library Python
pip install lmstudio
pip install python-Levenshtein
pip install tqdm atau pip install lmstudio Levenshtein tqdm

# Install LM Studio
1. Download LM Studio
2. Install aplikasi
3. Download model (qwen/qwen2.5-vl-7b)
4. Jalankan model tersebut di LM Studio.

# Start LM Studio
1. Buka LM Studio
2. Load model (qwen/qwen2.5-vl-7b)
3. Pastikan model berhasil dijalankan
4. Biarkan LM Studio tetap aktif selama proses inferensi berlangsung

# Format Dataset
Indonesian License Plate Recognition
│
├── images/
│      image001.jpg
│      image002.jpg
│
└── labels/
       image001.txt
       image002.txt

Contoh isi label

BP2407UZO

Nama file gambar dan label harus sama.


# Prompt
Model menerima prompt berikut:
    text
Read the Indonesian license plate in this image.
Return ONLY the plate number without explanation.

Prompt dibuat agar model hanya mengembalikan nomor plat tanpa penjelasan tambahan.


# Cara Menjalankan Program

Buka terminal pada folder project kemudian jalankan

```bash
python AAS_CV.py
```
Program akan otomatis:
- Mengecek dataset
- Membaca seluruh ground truth
- Memuat model dari LM Studio
- Mengirim seluruh gambar
- Melakukan OCR
- Menghitung CER
- Menyimpan hasil ke CSV

# Output
Hasil OCR akan tersimpan pada

result/
└── hasil_ocr.csv

Format file

| image | ground_truth | prediction | CER_score |
|--------|--------------|------------|-----------|

Contoh

| image | Ground Truth | Prediction | CER |
|--------|--------------|------------|------|
| test001.jpg | B9140BCD | BS140BCD | 12.50% |

# Character Error Rate (CER)
Performa OCR dievaluasi menggunakan Character Error Rate (CER).
Rumus:

CER = (S + D + I) / N

dengan:
- S = Substitution
- D = Deletion
- I = Insertion
- N = Jumlah karakter Ground Truth

Semakin kecil nilai CER menunjukkan performa OCR semakin baik.

CER = 0% berarti seluruh karakter berhasil dikenali dengan benar.

# Pipeline Program
License Plate Image
        │
        ▼
Prepare Image (Python)
        │
        ▼
LM Studio
(Qwen2.5-VL-7B)
        │
        ▼
Vision Language Model
        │
        ▼
Prediction
        │
        ▼
Ground Truth Comparison
        │
        ▼
Character Error Rate
        │
        ▼
CSV Result

# Technologies
- Python
- LM Studio
- Qwen2.5-VL-7B
- Vision Language Model (VLM)
- Levenshtein Distance
- Character Error Rate (CER)
- CSV
- tqdm

# Future Improvements
Pengembangan yang dapat dilakukan pada project ini antara lain:

- Menggunakan model Vision Language Model yang lebih besar
- Melakukan preprocessing citra sebelum inferensi
- Optimasi prompt (Prompt Engineering)
- Batch inference untuk mempercepat proses
- Menggunakan quantized model agar lebih ringan
- Membandingkan performa dengan OCR tradisional seperti EasyOCR atau Tesseract

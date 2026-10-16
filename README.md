# Offline AI Voice Transcription POC for OpenMRS / Bahmni

A lightweight, containerized, offline-capable microservice built with **Python (FastAPI)** and **OpenAI-Whisper** designed to provide secure, local speech-to-text transcription for healthcare environments.

## 🎯 Purpose & Big Picture

In many low-resource or rural clinical settings where OpenMRS and Bahmni are deployed, internet connectivity is intermittent or completely unavailable. This Proof of Concept (POC) ensures that clinical notes can be transcribed locally on-premise without relying on external cloud APIs, guaranteeing:
* **Strict Data Privacy:** Patient voice data never leaves the local hospital hardware.
* **Full Offline Capability:** Operates entirely air-gapped without external internet dependencies after initial setup.
* **Seamless Integration:** Exposes a simple REST API that electronic medical record (EMR) frontends can easily integrate with.

---

## 🛠️ Tech Stack

* **Framework:** FastAPI & Uvicorn (Python 3.11)
* **AI Engine:** OpenAI-Whisper (`openai-whisper`, PyTorch CPU)
* **Audio Processing:** FFmpeg
* **Containerization:** Docker & Docker Desktop (WSL 2 backend)

---

## 🚀 Getting Started & Installation

### Prerequisites
Make sure you have **Docker Desktop** installed and running on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/AI-voice-transcription-openMRS-POC.git](https://github.com/your-username/AI-voice-transcription-openMRS-POC.git)
cd AI-voice-transcription-openMRS-POC

## 📊 Evaluation Data & Acknowledgements

To benchmark the offline Whisper model's accuracy on medical terminology, this Proof of Concept uses the **United-Syn-Med** dataset hosted on Hugging Face.

@dataset{united_medsyn,
  title={United-MedSyn: Medical Speech Dataset for ASR},
  author={United We Care},
  year={2024},
  license={CC BY-SA 4.0}
}
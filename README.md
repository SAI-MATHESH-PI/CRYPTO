# CRYPTO
# 🔐 VHADS Pro v2.0 - Video HIGHT Attack Detection System

[![Deploy to Render](https://render.com/images/deploy-to-render.svg)](https://render.com/deploy?repo=vhads-pro)
[![Production Ready](https://img.shields.io/badge/Status-Production-brightgreen)](https://vhads-pro.onrender.app)
[![IEEE Conference](https://img.shields.io/badge/IEEE-Conference%20Demo-orange)](https://ieeexplore.ieee.org)

## 🎯 Research Overview

**VHADS Pro** implements real-time cryptanalysis of the **HIGHT block cipher** (TTAS.KO-12.0042) using 64×64 CCTV frame analysis. The system detects 4 classes of side-channel attacks:

| Attack Type | Description | Prevalence |
|-------------|-------------|------------|
| **Normal** | Clean encryption | 25% |
| **Fault** | Transient bit flips | 25% |
| **Reduced** | Early round termination | 25% |
| **Differential** | Controlled differences | 25% |

**Performance**: Macro-F1 = **49.8%**, ROC-AUC = **50.5%** (15K training frames)

## 🏗️ Production Architecture


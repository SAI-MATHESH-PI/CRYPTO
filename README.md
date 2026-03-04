# CRYPTO
# 🔐 VHADS Pro v2.0 - Video HIGHT Attack Detection System

[![Deploy to Render](https://render.com/images/deploy-to-render.svg)](https://render.com/deploy?repo=vhads-pro)
[![Production Ready](https://img.shields.io/badge/Status-Production-brightgreen)](https://vhads-pro.onrender.app)
[![IEEE Conference](https://img.shields.io/badge/IEEE-Conference%20Demo-orange)](https://ieeexplore.ieee.org)

## 🎯 Research Overview

**VHADS Pro** implements real-time cryptanalysis of the **HIGHT block cipher** (TTAS.KO-12.0042) using 64×64 CCTV frame analysis. The system detects 4 classes of side-channel attacks:

| Attack Type | Description | Prevalence |
|-------------|-------------|------------|
| **Normal** | Clean encryption | 70% |
| **Fault** | Transient bit flips | 15% |
| **Reduced** | Early round termination | 10% |
| **Differential** | Controlled differences | 5% |

**Performance**: Macro-F1 = **89.8%**, ROC-AUC = **94.2%** (15K training frames)

## 🏗️ Production Architecture


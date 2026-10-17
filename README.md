# 🩺 VirusDiag — Système de Diagnostic Viral par IA & POO

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-005571.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **VirusDiag** est un projet académique d'application avancée combinant le **Machine Learning from scratch** (via `numpy`), une architecture logicielle rigoureuse basée sur la **Programmation Orientée Objet (POO)**, une API moderne (`FastAPI`) et une interface clinique interactive (`Streamlit`).

---

## 📑 Table des matières
1. [À propos du projet](#-à-propos-du-projet)
2. [Arborescence du projet](#-arborescence-du-projet)
3. [Fonctionnalités principales](#-fonctionnalités-principales)
4. [Installation & Prérequis](#-installation--prérequis)
5. [Guide d'utilisation](#-guide-dutilisaton)
6. [Concepts POO illustrés](#-concepts-poo-illustrés)
7. [Tests unitaires](#-tests-unitaires)
8. [Licence](#-licence)

---

## 🎯 À propos du projet

Le projet simule un système d'aide à la décision médicale permettant de classifier l'état d'un patient (**"Infecté"** ou **"Sain"**) en fonction de paramètres cliniques (température, saturation en oxygène, globules blancs, symptômes, etc.).

L'accent est mis sur les bonnes pratiques de génie logiciel :
* **Principe de Responsabilité Unique (SRP)** : Séparation stricte entre l'entraînement (`Trainer`), l'évaluation (`Evaluator`), la prédiction clinique (`ClinicalPredictor`) et la persistance des données.
* **Polymorphisme et Abstraction** : Les modèles de Machine Learning (`LogisticRegression`, `NeuralNetwork`) héritent tous d'une classe abstraite commune (`Model`), garantissant l'interchangeabilité totale des algorithmes.

---

## 📂 Arborescence du projet

```text
virus_diag/
├── data/
│   ├── generate_data.py       # Script de génération du dataset synthétique
│   └── patient_data.csv       # Données cliniques des patients
├── core/
│   ├── dataset.py             # Chargement, nettoyage et train/test split
│   ├── model.py               # Classe abstraite de base (ABC) pour les modèles
│   ├── logistic_regression.py # Implémentation d'une régression logistique sur mesure
│   ├── neural_network.py      # Implémentation d'un perceptron multicouche simple
│   └── optimizer.py           # Algorithmes d'optimisation (SGD, Adam simplifié)
├── pipeline/
│   ├── trainer.py             # Orchestration de l'entraînement et normalisation
│   └── evaluator.py           # Calcul des métriques d'évaluation
├── utils/
│   ├── preprocessing.py       # Normalisation des caractéristiques (StandardScaler)
│   └── metrics.py             # Accuracy, Précision, Rappel, F1-Score, Matrice de confusion
├── app/
│   ├── api.py                 # API REST FastAPI exposant le point de terminaison /predict
│   └── interface_clinique.py  # Classe ClinicalPredictor & Interface graphique Streamlit
├── tests/
│   └── test_virus_diag.py     # Tests unitaires (pytest)
├── requirements.txt           # Dépendances du projet
└── README.md                  # Documentation du projet
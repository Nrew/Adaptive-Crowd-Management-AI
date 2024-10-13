# Adaptive AI for Emergency Crowd Management in Real-time Virtual Environments

[![Unity Build and Test](https://github.com/Nrew/Adaptive-Crowd-Management-AI/actions/workflows/unity_ci.yml/badge.svg)](https://github.com/Nrew/Adaptive-Crowd-Management-AI/actions/workflows/unity_ci.yml)
[![Python Lint and Test](https://github.com/Nrew/Adaptive-Crowd-Management-AI/actions/workflows/python_ci.yml/badge.svg)](https://github.com/Nrew/Adaptive-Crowd-Management-AI/actions/workflows/python_ci.yml)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Team Members](#team-members)
- [License](#license)

## Project Overview

An AI-based system designed to model, simulate, and predict crowd behavior in emergency situations within virtual environments. The system leverages multi-agent simulations, adaptive pathfinding, emotion analysis, and human-in-the-loop interventions to optimize emergency responses.

## Features

- **Multi-Agent Simulation Framework**
- **Adaptive Pathfinding Algorithms**
- **Emotion and Panic Level Analysis Using Neural Networks**
- **Interactive Human-in-the-Loop Dashboard**
- **Data Visualization and Post-Simulation Analytics**

## Technologies Used

- **Unity Engine** for simulation visualization
- **Python** with TensorFlow/PyTorch for AI models
- **Reinforcement Learning** techniques
- **Neural Networks** for emotion analysis
- **Flask/Django** for the web dashboard
- **D3.js** for data visualization

## Project Structure

```plaintext
Adaptive-Crowd-Management-AI/
├── unity_simulation/            # Unity project files
├── ai_models/                   # AI models and training scripts
├── dashboard/                   # Web dashboard application
├── data_visualization/          # Visualization scripts and data
├── docs/                        # Documentation and manuals
├── .github/workflows/           # GitHub Actions workflows
├── README.md                    # Project README file
├── LICENSE                      # License information
└── .gitignore                   # Git ignore rules
```
## Setup and Installation
### **Prerequisites**
- **Unity Engine** (version 2022.3 v5 LTS)
- **Python** (version 3.11 or higher)
- **Git** for version control

### Installation Steps
1. **Clone the Repository**:
  ```bash
  [git clone https://github.com/Nrew/Adaptive-Crowd-Management-AI.git]
  ```
2. **Set Up Unity Simulation**:
  - Open Unity Hub.
  - Click "**Add**" and select the ``unity_simulation/`` folder
  - Open the project in Unity.
3. **Set Up Python Enviroment**:
  - Navigate to the ``ai_models/`` directory.
  - Create and activate a virtual enviroment:
  ```bash
  python -m venv {env_name}
  source {env_name}/bin/activate # Windows: {env_name}\Scripts\activate
  ```
  - Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
4. **Set Up Dashboard Application**:
  - Navigate to the ``dashboard/`` directory.
  - Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
## Usage
### **Running the Unity Simulation**
- Open the Unity Project.
- Press the "**Play**" button to start the simulation.
- Adjust paramteres via the inspector or UI as needed.

### **Training AI Models**
- Ensure your virtual enviroment is activated.
- Run training scripts in the ``ai_models/`` directory:
  ```bash
  python reinforcement_learning/train_agent.py
  ```
## **Starting the Dashboard**
- Navigate to the ``dashboard/`` directory.
- Run the application:
  ```bash
  python app.py
  ```
- Access the dashboard at ``http://localhost:{port_num}`` in your web browser.
## **Data Visualization**
- Use the scripts in ``data_visualization/`` to generate visual reports.
- Example:
  ```bash
  python visualizations/generate_heatmap.py
  ```
### Contributing
**N/A at this time until 2025.**

We welcome contributions! Please read our Contributing Guidelines for details on our code of conduct, and the process for submitting pull requests.

### Team Members
- Andrew Sayegh
- Daniel Tkachov
- Adrian Halgas
- Geeta Venkata Siva Karthik Kasaraneni
- Radhika Khurana

## License
This project is licensed under the terms of the [MIT License](LICENSE).

# FuzzyLight: Fuzzy Logic-based Traffic Light Control System

FuzzyLight is an intelligent traffic signal control system that combines fuzzy logic for phase selection with deep reinforcement learning (DDPG) for optimizing action duration. The system is designed to reduce traffic congestion and improve intersection efficiency.

## Overview

This project implements a novel traffic light control algorithm that uses:
- **Fuzzy Logic**: For intelligent phase selection based on traffic flow patterns
- **Deep Reinforcement Learning (DDPG)**: For optimizing signal duration timing
- **Multi-Head Attention**: For enhanced lane feature embedding and phase representation
- **CityFlow Simulator**: For realistic traffic simulation and evaluation

## Features

- Supports multiple city road networks (Hangzhou, Jinan, New York)
- Adaptive signal timing based on real-time traffic conditions
- Multi-intersection coordination support
- Configurable traffic scenarios and road networks
- Comprehensive performance evaluation and visualization

## Project Structure

```
FuzzyLight/
├── models/              # Neural network models
│   ├── fuzzy_light.py  # Main FuzzyLight agent (with attention)
│   ├── fuzzy_light_with_no_attention.py  # Variant without attention
│   ├── agent.py        # Base agent class
│   └── network_agent.py # Network agent implementation
├── utils/               # Utility functions and tools
│   ├── config.py       # Configuration settings
│   ├── pipeline.py     # Training pipeline
│   ├── cityflow_env.py # CityFlow environment wrapper
│   └── ...             # Other utilities
├── data/               # Traffic data and road networks
│   ├── Hangzhou/       # Hangzhou traffic scenarios
│   ├── Jinan/          # Jinan traffic scenarios
│   └── newyork_28_7/   # New York traffic scenarios
├── run_fuzzylight.py   # Main execution script
└── video.mp4           # Comparison video

```

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy
- CityFlow (traffic simulator)

## Usage

### Basic Usage

Run the FuzzyLight algorithm with default settings:

```bash
python run_fuzzylight.py
```

### Command Line Options

The script supports various command-line arguments:

```bash
python run_fuzzylight.py \
    -memo <memo_name> \           # Experiment name/memo
    -mod <model_name> \            # Model name (e.g., "FuzzyLight", "DirectLight")
    -gen <num_generators> \        # Number of generators (default: 1)
    -multi_process \               # Enable multi-processing
    -workers <num_workers> \       # Number of worker processes
    -hangzhou \                    # Use Hangzhou road network
    -jinan \                       # Use Jinan road network
    -newyork                       # Use New York road network
```

### Example

Run with Hangzhou road network:

```bash
python run_fuzzylight.py -hangzhou -memo fuzzy_light_exp -mod FuzzyLight
```

## Algorithm Description

### Fuzzy Logic Phase Selection

The system uses fuzzy logic to select the optimal traffic phase based on lane queue lengths. It aggregates vehicle counts from different directions and selects the phase with the highest traffic demand.

### Deep Reinforcement Learning

The DDPG (Deep Deterministic Policy Gradient) algorithm is used to learn optimal signal duration:
- **Actor Network**: Predicts continuous action values (signal duration)
- **Critic Network**: Evaluates the quality of actions
- **Multi-Head Attention**: Enhances lane feature representation
- **Ornstein-Uhlenbeck Noise**: For exploration during training

### Key Components

1. **Phase Control Policy**: Fuzzy logic-based phase selection
2. **Lane Embedding**: Multi-head attention mechanism for lane feature extraction
3. **Actor-Critic Architecture**: DDPG implementation for duration optimization
4. **Memory Buffer**: Experience replay for training stability

## Configuration

Key configuration parameters can be modified in `utils/config.py`:

- `NUM_PHASES`: Number of traffic phases (default: 4)
- `MAX_LANE`: Maximum number of lanes (default: 12)
- `ACTION_DURATION`: Valid signal duration options
- `DIC_REWARD_INFO`: Reward function weights
- Learning rates, discount factors, and other hyperparameters

## Results

The included `video.mp4` compares traffic light operation with and without the FuzzyLight algorithm, demonstrating improvements in traffic flow and reduced queue lengths.

## Citation

If you use this code in your research, please cite the associated paper.

## License

[Specify license here]

## Contact

[Add contact information if needed]

---
title: DevToolKit
emoji: 👁
colorFrom: purple
colorTo: pink
language: py
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
license: apache-2.0
port: 8888
models:
  - gemini-pro-2.0
  - Salesforce/codet5-small
  - bigscience/T0_3B
---

[![Hugging Face Spaces - SDK](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20SDK-blue)](https://huggingface.co/spaces/whackthejacker/whackthejacker/DevToolKit)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-orange.svg)](https://streamlit.io/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Models: Gemini Pro 2.0, CodeT5-small, T0_3B](https://img.shields.io/badge/Models-Gemini%20Pro%202.0%2C%20CodeT5--small%2C%20T0_3B-green)](https://huggingface.co/whackthejacker/ensemble-model-app-builder)

# DevToolKit

[![GitHub license](https://img.shields.io/github/license/whackthejacker/DevToolKit?style=flat-square)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/whackthejacker/DevToolKit?style=flat-square)](https://github.com/whackthejacker/DevToolKit/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/whackthejacker/DevToolKit?style=flat-square)](https://github.com/whackthejacker/DevToolKit/network)
[![Hugging Face Spaces](https://img.shields.io/badge/HF%20Spaces-DevToolKit-blue?style=flat-square)](https://huggingface.co/spaces/whackthejacker/DevToolKit)

_A one-stop toolkit for developers, powered by Hugging Face Spaces_

## Meta Description

DevToolKit is an open-source, multi-functional development toolkit designed to streamline daily coding tasks and enhance productivity. Hosted on Hugging Face Spaces, this project duplicates and builds upon the innovative work by acecalisto3, providing a clean and user-friendly interface to access a suite of developer tools.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation & Usage](#installation--usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

DevToolKit brings together a collection of developer utilities into one accessible space. Whether you need to encode data, perform quick conversions, or simply enhance your productivity, DevToolKit provides the tools you need—all in a single, easy-to-use interface.

This project is hosted as a [Hugging Face Space](https://huggingface.co/spaces/whackthejacker/DevToolKit) and is a duplicate of [acecalisto3/DevToolKit](https://huggingface.co/spaces/acecalisto3/DevToolKit), improved and maintained by [whackthejacker](https://huggingface.co/whackthejacker).

## Features

- **Multi-Tool Integration:** Access various developer utilities from one place.
- **Web Interface:** Run the toolkit directly in your browser via Hugging Face Spaces.
- **Extensible Design:** Easily add or remove tools to suit your workflow.
- **Responsive & Clean UI:** Designed with a focus on simplicity and ease of use.
- **Open Source:** Contributions are welcome to enhance functionality and add new features.

## Installation & Usage

### Running on Hugging Face Spaces

DevToolKit is already deployed on Hugging Face Spaces. To use the toolkit, simply navigate to:

[https://huggingface.co/spaces/whackthejacker/DevToolKit](https://huggingface.co/spaces/whackthejacker/DevToolKit)

### Running Locally

If you wish to run DevToolKit locally:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/whackthejacker/DevToolKit.git
    cd DevToolKit
    ```

2. **Install dependencies:**

    Ensure you have [Docker](https://www.docker.com/) installed. Then build and run the Docker container:

    ```bash
    docker build -t devtoolkit .
    docker run -p 7860:7860 devtoolkit
    ```

3. **Access the Application:**

    Open your browser and navigate to [http://localhost:7860](http://localhost:7860).

## Customization

DevToolKit is designed to be flexible. To add your own tools or modify existing ones:

- **Tool Integration:** Modify the `tools` directory (or the corresponding configuration file) to add new utilities.
- **UI Customization:** Update the frontend components to match your desired look and feel.
- **Configuration:** Edit the configuration files as needed to set parameters, adjust routing, or change display options.

For detailed customization instructions, please refer to the [Contributing Guidelines](CONTRIBUTING.md).

## Contributing

Contributions are very welcome! If you have ideas, bug fixes, or new features to add, please:

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a clear description of your changes.

For more details, please see our [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or suggestions, please contact:

- **Author:** [whackthejacker](https://huggingface.co/whackthejacker)
- **Issue Tracker:** [GitHub Issues](https://github.com/whackthejacker/DevToolKit/issues)

---

*Happy Coding!*